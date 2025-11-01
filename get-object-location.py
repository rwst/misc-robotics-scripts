import argparse
import os
import sys
import numpy as np
import cv2
import trimesh
import google.generativeai as genai
from PIL import Image
import io
import re
import math

# --- Configuration ---
# It's recommended to install the contrib version for SIFT
# pip install opencv-contrib-python
try:
    # Check if SIFT is available, often in the contrib module
    cv2.SIFT_create()
except AttributeError:
    print("SIFT not found. Please install 'opencv-contrib-python'.")
    sys.exit(1)

# Configure the Gemini API key
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise KeyError
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

# --- Helper Functions ---

def get_bounding_box(image_path, text_description):
    """
    Uses the Gemini API to get a bounding box for a described object in an image.

    Args:
        image_path (str): The path to the image file.
        text_description (str): The text describing the object to find.

    Returns:
        tuple: A tuple (x_center, y_center, width, height) of the bounding box
               in pixel coordinates, or None if not found.
    """
    print(f"Attempting to find '{text_description}' in {image_path} using Gemini API...")
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.5-flash-image')

        prompt = (f"Return a bounding box for '{text_description}' in this image "
                  "in the format [ymin, xmin, ymax, xmax]. "
                  "Provide only the list of numbers.")

        response = model.generate_content([prompt, img])
        
        # Use regex to find the list-like structure in the response text
        match = re.search(r'\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', response.parts[0].text)
        if not match:
            print(f"Warning: Could not parse bounding box from Gemini response: {response.parts[0].text}")
            return None

        coords = [int(c) for c in match.groups()]
        
        # The model returns normalized coordinates scaled by 1000
        img_width, img_height = img.size
        ymin = (coords[0] / 1000) * img_height
        xmin = (coords[1] / 1000) * img_width
        ymax = (coords[2] / 1000) * img_height
        xmax = (coords[3] / 1000) * img_width

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        print(f"Found bounding box for '{text_description}' at center=({x_center:.2f}, {y_center:.2f})")
        return (x_center, y_center, width, height)

    except Exception as e:
        print(f"An error occurred while using the Gemini API: {e}")
        return None

def describe_image(image_path, text_description):
    """
    Uses the Gemini API to describe an image based on a text prompt.

    Args:
        image_path (str): The path to the image file.
        text_description (str): The text prompt for the description.
    """
    print(f"Attempting to describe '{image_path}' with prompt: '{text_description}'")
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel('gemini-2.5-flash-image')
        response = model.generate_content([text_description, img])
        print("--- Gemini API Response ---")
        print(response.parts[0].text)
        print("---------------------------")
    except Exception as e:
        print(f"An error occurred while using the Gemini API: {e}")



def estimate_camera_intrinsics(image_size):
    """
    Creates an estimated camera intrinsic matrix based on image dimensions.
    This is a simplification for when camera calibration data is not available.

    Args:
        image_size (tuple): A tuple (width, height) of the image.

    Returns:
        numpy.ndarray: The 3x3 camera intrinsic matrix.
    """
    width, height = image_size
    focal_length = width  # A common heuristic
    center_x = width / 2
    center_y = height / 2
    
    cam_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    return cam_matrix

def find_3d_2d_correspondences(image, stl_mesh, robot_bbox):
    """
    Finds correspondences between 3D points from an STL mesh and 2D points in an image.

    This function uses a simplified approach by projecting STL vertices to a canonical
    2D view to generate descriptors, which are then matched with the input image.

    Args:
        image (numpy.ndarray): The input image (read by OpenCV).
        stl_mesh (trimesh.Trimesh): The loaded STL mesh of the robot arm.
        robot_bbox (tuple): The bounding box of the robot in the image to focus the search.

    Returns:
        tuple: A tuple containing (list of 3D points, list of corresponding 2D points).
    """
    print("Finding 3D-2D correspondences...")
    sift = cv2.SIFT_create()

    # 1. Extract features from the real image within the robot's bounding box
    x_center, y_center, width, height = robot_bbox
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    robot_roi = image[y1:y2, x1:x2]
    
    kp_image, desc_image = sift.detectAndCompute(robot_roi, None)
    if desc_image is None:
        print("Warning: No SIFT features found in the robot ROI of the image.")
        return [], []
        
    # Adjust keypoint coordinates to be relative to the full image
    for kp in kp_image:
        kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)

    # 2. For simplicity, we'll use the mesh vertices as our 3D points.
    # A more robust solution would involve rendering and feature extraction from the render.
    # Here, we will use a simplified feature matching against a projection.
    
    # Let's project the 3D vertices onto a 2D plane to create a reference "image"
    # This is a significant simplification.
    points_3d = stl_mesh.vertices
    x_coords = points_3d[:, 0]
    y_coords = points_3d[:, 1]
    
    # Normalize to create a pseudo-image
    x_norm = cv2.normalize(x_coords, None, 0, 512, cv2.NORM_MINMAX, cv2.CV_8U)
    y_norm = cv2.normalize(y_coords, None, 0, 512, cv2.NORM_MINMAX, cv2.CV_8U)
    
    ref_img = np.zeros((512, 512), dtype=np.uint8)
    kp_ref = []
    for i, (x, y) in enumerate(zip(x_norm, y_norm)):
        ref_img[y[0], x[0]] = 255
        kp_ref.append(cv2.KeyPoint(x=x[0], y=y[0], size=5, _angle=0, _response=0, _octave=0, _class_id=i))

    kp_ref, desc_ref = sift.compute(ref_img, kp_ref)
    if desc_ref is None:
        print("Warning: Could not compute SIFT descriptors for the reference STL projection.")
        return [], []

    # 3. Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc_ref, desc_image)
    matches = sorted(matches, key=lambda x: x.distance)

    # 4. Extract corresponding points from good matches
    obj_points_3d = []
    img_points_2d = []
    for match in matches[:50]: # Use top 50 matches
        ref_idx = match.queryIdx
        img_idx = match.trainIdx
        
        original_3d_point_index = kp_ref[ref_idx].class_id
        obj_points_3d.append(points_3d[original_3d_point_index])
        img_points_2d.append(kp_image[img_idx].pt)

    print(f"Found {len(obj_points_3d)} potential correspondences.")
    return np.array(obj_points_3d, dtype=np.float32), np.array(img_points_2d, dtype=np.float32)

def estimate_pose(points_3d, points_2d, camera_matrix):
    """
    Estimates camera pose using solvePnP.

    Args:
        points_3d (numpy.ndarray): Array of 3D points.
        points_2d (numpy.ndarray): Array of corresponding 2D points.
        camera_matrix (numpy.ndarray): The camera intrinsic matrix.

    Returns:
        tuple: A tuple (rotation_vector, translation_vector) or (None, None).
    """
    if len(points_3d) < 4:
        print("Error: Need at least 4 point correspondences for solvePnP.")
        return None, None
        
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    try:
        success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs)
        if success:
            print("Pose estimation successful.")
            return rvec, tvec
        else:
            print("Warning: solvePnP failed.")
            return None, None
    except cv2.error as e:
        print(f"Error during solvePnP: {e}")
        return None, None


def triangulate_3d_point(point1, point2, rvec1, tvec1, rvec2, tvec2, camera_matrix):
    """
    Calculates the 3D position of a point using triangulation.

    Args:
        point1 (tuple): The (x, y) coordinates of the point in the first image.
        point2 (tuple): The (x, y) coordinates of the point in the second image.
        rvec1, tvec1: Pose of the first camera.
        rvec2, tvec2: Pose of the second camera.
        camera_matrix: The camera intrinsic matrix.

    Returns:
        numpy.ndarray: The triangulated 3D point (x, y, z), or None.
    """
    print("Triangulating cuboid's 3D position...")
    # Convert rotation vectors to rotation matrices
    rmat1, _ = cv2.Rodrigues(rvec1)
    rmat2, _ = cv2.Rodrigues(rvec2)

    # Create projection matrices
    proj_matrix1 = camera_matrix @ np.hstack((rmat1, tvec1))
    proj_matrix2 = camera_matrix @ np.hstack((rmat2, tvec2))

    # Points must be in a specific format for triangulatePoints
    p1 = np.array([point1], dtype=np.float32).T
    p2 = np.array([point2], dtype=np.float32).T

    # Triangulate
    points_4d_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, p1, p2)
    
    # Convert from homogeneous to Cartesian coordinates
    if points_4d_hom[3] != 0:
        points_3d = points_4d_hom[:3] / points_4d_hom[3]
        return points_3d.flatten()
    else:
        print("Warning: Triangulation failed (division by zero).")
        return None

def ray_plane_intersection(rvec, tvec, camera_matrix, point_2d):
    """
    Calculates the intersection of a ray from the camera with the z=0 plane.

    Args:
        rvec, tvec: Pose of the camera.
        camera_matrix: The camera intrinsic matrix.
        point_2d (tuple): The (x, y) pixel coordinate of the object.

    Returns:
        numpy.ndarray: The 3D intersection point (x, y, z), or None.
    """
    print("Calculating cuboid position via ray-plane intersection...")
    # 1. Convert camera pose to world coordinates
    rmat, _ = cv2.Rodrigues(rvec)
    # The camera's position in the world frame is -R' * T
    cam_pos = -np.matrix(rmat).T * np.matrix(tvec)
    
    # 2. Unproject 2D point to a 3D ray in camera coordinates
    point_3d_cam = np.linalg.inv(camera_matrix) @ np.array([point_2d[0], point_2d[1], 1.0])

    # 3. Transform the ray direction vector to world coordinates
    ray_dir_world = np.matrix(rmat).T @ np.matrix(point_3d_cam).T

    # 4. Define the plane (z=0)
    plane_normal = np.array([0, 0, 1])
    plane_point = np.array([0, 0, 0])
    
    # 5. Calculate intersection
    # Based on the formula: t = ((plane_point - ray_origin) . plane_normal) / (ray_direction . plane_normal)
    ray_origin = np.array(cam_pos).flatten()
    ray_direction = np.array(ray_dir_world).flatten()
    
    denom = np.dot(ray_direction, plane_normal)
    if abs(denom) > 1e-6:
        t = np.dot(plane_point - ray_origin, plane_normal) / denom
        intersection_point = ray_origin + t * ray_direction
        return intersection_point
    else:
        print("Warning: Ray is parallel to the plane. No intersection.")
        return None

# --- Main Logic ---

def main():
    """Main function to parse arguments and run the localization process."""
    parser = argparse.ArgumentParser(
        description="Determine the relative location of a textually described object "
                    "with respect to a robot arm from video frames."
    )
    parser.add_argument("--image1", required=True, help="Path to the first video frame image.")
    parser.add_argument("--image2", help="Path to the second video frame image (for two-frame method).")
    parser.add_argument("--stl", help="Path to the STL file of the robot arm.")
    parser.add_argument("--object_description", help="Textual description of the target object.")
    parser.add_argument("--robot_description", help="Textual description of the robot arm.")
    parser.add_argument("--describe", action="store_true", help="Describe the image instead of finding an object.")
    
    args = parser.parse_args()

    if args.describe:
        if not args.object_description:
            print("Error: --object_description is required for --describe mode.")
            return
        describe_image(args.image1, args.object_description)
        return

    # --- Step 1: Object Identification using Visual Grounding ---
    
    # Get bounding box for the object in the first image
    obj_bbox1 = get_bounding_box(args.image1, args.object_description)
    if not obj_bbox1:
        print(f"Could not identify '{args.object_description}' in {args.image1}. Exiting.")
        return
    obj_center1 = (obj_bbox1[0], obj_bbox1[1])

    # Get bounding box for the robot arm in the first image
    robot_bbox1 = get_bounding_box(args.image1, args.robot_description)
    if not robot_bbox1:
        print(f"Could not identify '{args.robot_description}' in {args.image1}. Exiting.")
        return

    # Load images and STL
    try:
        image1 = cv2.imread(args.image1)
        if image1 is None: raise FileNotFoundError
        stl_mesh = trimesh.load_mesh(args.stl)
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading files: {e}")
        return

    # Assume camera intrinsics (a major simplification)
    cam_matrix = estimate_camera_intrinsics((image1.shape[1], image1.shape[0]))
    
    # --- Determine Method: Single or Two-Frame ---
    
    if args.image2:
        # --- Method A: Two-Frame Approach ---
        print("\n--- Starting Two-Frame Method (Higher Accuracy) ---")
        
        # Load second image
        try:
            image2 = cv2.imread(args.image2)
            if image2 is None: raise FileNotFoundError
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading second image: {e}")
            return
            
        # Get object and robot bounding boxes for the second image
        obj_bbox2 = get_bounding_box(args.image2, args.object_description)
        if not obj_bbox2:
            print(f"Could not identify '{args.object_description}' in {args.image2}. Exiting.")
            return
        obj_center2 = (obj_bbox2[0], obj_bbox2[1])

        robot_bbox2 = get_bounding_box(args.image2, args.robot_description)
        if not robot_bbox2:
            print(f"Could not identify '{args.robot_description}' in {args.image2}. Exiting.")
            return

        # Step 2a & 2b for Image 1: Correspondences and Pose Estimation
        points_3d_1, points_2d_1 = find_3d_2d_correspondences(image1, stl_mesh, robot_bbox1)
        rvec1, tvec1 = estimate_pose(points_3d_1, points_2d_1, cam_matrix)
        if rvec1 is None:
            print("Failed to estimate camera pose for image 1. Cannot proceed with two-frame method.")
            return

        # Step 2a & 2b for Image 2: Correspondences and Pose Estimation
        points_3d_2, points_2d_2 = find_3d_2d_correspondences(image2, stl_mesh, robot_bbox2)
        rvec2, tvec2 = estimate_pose(points_3d_2, points_2d_2, cam_matrix)
        if rvec2 is None:
            print("Failed to estimate camera pose for image 2. Cannot proceed with two-frame method.")
            return

        # Step 2c: Triangulate the Cuboid's 3D Position
        cuboid_pos_3d = triangulate_3d_point(obj_center1, obj_center2, rvec1, tvec1, rvec2, tvec2, cam_matrix)
        if cuboid_pos_3d is None:
            print("Failed to determine cuboid position through triangulation.")
            return

    else:
        # --- Method B: Single-Frame Approach ---
        print("\n--- Starting Single-Frame Method (Higher Assumption) ---")
        
        # Step 2a: Estimate Camera Pose
        points_3d, points_2d = find_3d_2d_correspondences(image1, stl_mesh, robot_bbox1)
        rvec, tvec = estimate_pose(points_3d, points_2d, cam_matrix)
        if rvec is None:
            print("Failed to estimate camera pose. Cannot proceed.")
            return

        # Step 2c: Localize via Ray-Plane Intersection
        cuboid_pos_3d = ray_plane_intersection(rvec, tvec, cam_matrix, obj_center1)
        if cuboid_pos_3d is None:
            print("Failed to determine cuboid position through ray-plane intersection.")
            return

    # --- Step 2d: Calculate and Display Relative Position ---
    print("\n--- Results ---")
    print(f"Calculated 3D position of the object's center: "
          f"x={cuboid_pos_3d[0]:.4f}, y={cuboid_pos_3d[1]:.4f}, z={cuboid_pos_3d[2]:.4f}")

    x, y = cuboid_pos_3d[0], cuboid_pos_3d[1]
    
    # Cartesian coordinates (relative to the robot base at (0,0))
    print(f"\nRelative Cartesian Position (x, y): ({x:.4f}, {y:.4f})")

    # Polar coordinates
    r = math.sqrt(x**2 + y**2)
    phi_rad = math.atan2(y, x)
    phi_deg = math.degrees(phi_rad)
    
    print(f"Relative Polar Position (r, φ): (r={r:.4f}, φ={phi_deg:.2f}°)")


if __name__ == "__main__":
    main()
