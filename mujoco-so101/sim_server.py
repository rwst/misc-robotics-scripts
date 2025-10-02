import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from PIL import Image


class SO101Env(MujocoEnv):
    """
    Custom MuJoCo environment for the SO101 robot.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
    }

    def __init__(self, model_path, frame_skip=1, **kwargs):
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=None,  # Will be set below
            **kwargs,
        )

        # The observation space consists of joint positions and velocities.
        # The size is determined by the model.
        obs_size = self.data.qpos.size + self.data.qvel.size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def _get_obs(self):
        """
        Returns the observation from the environment.
        """
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def reset_model(self):
        """
        Resets the robot to a neutral starting position.
        """
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        """
        Applies an action to the environment.
        """
        self.data.ctrl[:action.shape[0]] = action
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        observation = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

def check_aabb_overlap(pos1_xy, size1_wh, pos2_xy, size2_wh):
    """
    Checks for Axis-Aligned Bounding Box (AABB) overlap between two rectangles
    defined by their center positions and full sizes (width, height).
    """
    half_s1_x, half_s1_y = size1_wh[0] / 2, size1_wh[1] / 2
    half_s2_x, half_s2_y = size2_wh[0] / 2, size2_wh[1] / 2

    min_x1, max_x1 = pos1_xy[0] - half_s1_x, pos1_xy[0] + half_s1_x
    min_y1, max_y1 = pos1_xy[1] - half_s1_y, pos1_xy[1] + half_s1_y
    min_x2, max_x2 = pos2_xy[0] - half_s2_x, pos2_xy[0] + half_s2_x
    min_y2, max_y2 = pos2_xy[1] - half_s2_y, pos2_xy[1] + half_s2_y

    return (max_x1 > min_x2 and min_x1 < max_x2 and
            max_y1 > min_y2 and min_y1 < max_y2)

def get_cam_orientation(cam_pos, target_pos):
    """
    Computes the quaternion orientation for the camera to look towards the target position.
    
    :param cam_pos: numpy array of shape (3,) for camera position [x, y, z]
    :param target_pos: numpy array of shape (3,) for target position [x, y, z]
    :return: numpy array of shape (4,) for quaternion [w, x, y, z]
    """
    fwd_dir = target_pos - cam_pos
    fwd_dir = fwd_dir / np.linalg.norm(fwd_dir)
    
    Z = -fwd_dir
    world_up = np.array([0.0, 0.0, 1.0])
    
    X = np.cross(world_up, Z)
    X = X / np.linalg.norm(X)
    
    Y = np.cross(Z, X)
    
    mat = np.column_stack((X, Y, Z))
    
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    
    return quat

def randomize(env, yellow_adr, blue_adr, side_camera_id, top_camera_id):
    """
    Randomizes the positions and orientations of the yellow cube and blue tray
    within a defined sandbox, checking for overlaps, and adjusts the side camera.
    """
    # Constants from the problem description
    d = 0.4
    A_w, A_h = 0.05, 0.05  # Arm base dimensions (full width, height)
    s_w, s_h = 0.02, 0.01  # Yellow cube dimensions (full width, height)
    S_w, S_h = 0.1, 0.15  # Blue tray dimensions (full width, height)

    # Sandbox dimensions based on document defaults
    # Arm is at (A_x, A_y) = (0,0), O_w = d, O_h = 2d
    O_w = d
    O_h = 2 * d
    # lower left corner of sandbox is (O_x=0, O_y=-O_h/2)
    O_x = 0
    O_y = -O_h / 2

    # Arm base position (center)
    A_x, A_y = 0, 0
    arm_base_pos_xy = np.array([A_x, A_y])
    arm_base_size_wh = np.array([A_w, A_h])

    # Object Z-heights from XML (fixed, assuming they are consistent)
    S_z = 0.01  # Blue tray Z position
    s_z = 0.025 # Yellow cube Z position

    tray_placed = False
    while not tray_placed:
        # 2. compute random tray position (S_x,S_y) within sandbox with random orientation Sigma
        # Calculate bounds for tray center to ensure its center is within sandbox
        tray_min_center_x = O_x
        tray_max_center_x = O_x + O_w
        tray_min_center_y = O_y
        tray_max_center_y = O_y + O_h

        S_x = np.random.uniform(tray_min_center_x, tray_max_center_x)
        S_y = np.random.uniform(tray_min_center_y, tray_max_center_y)
        Sigma = np.random.uniform(-np.pi, np.pi)  # Random orientation (radians)

        tray_pos_xy = np.array([S_x, S_y])
        tray_size_wh = np.array([S_w, S_h]) # Using nominal size for AABB overlap check

        # 3. if tray overlaps arm base, reset and go to 1
        if check_aabb_overlap(tray_pos_xy, tray_size_wh, arm_base_pos_xy, arm_base_size_wh):
            continue # Retry tray placement

        # Store tentative tray placement data (x, y, z, qw, qx, qy, qz)
        qpos_tray_rot = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qpos_tray_rot, np.array([0, 0, 1]), Sigma) # Rotate around Z-axis
        blue_tray_qpos = np.array([S_x, S_y, S_z, qpos_tray_rot[0], qpos_tray_rot[1], qpos_tray_rot[2], qpos_tray_rot[3]])

        cube_placed = False
        while not cube_placed:
            # 4. compute random cube position (s_x,s_y) within sandbox with random orientation sigma
            cube_min_center_x = O_x
            cube_max_center_x = O_x + O_w
            cube_min_center_y = O_y
            cube_max_center_y = O_y + O_h

            s_x = np.random.uniform(cube_min_center_x, cube_max_center_x)
            s_y = np.random.uniform(cube_min_center_y, cube_max_center_y)
            sigma = np.random.uniform(-np.pi, np.pi)  # Random orientation (radians)

            cube_pos_xy = np.array([s_x, s_y])
            cube_size_wh = np.array([s_w, s_h]) # Using nominal size for AABB overlap check

            # 5. if cube overlaps tray or arm base, go to 3 (reset tray)
            if check_aabb_overlap(cube_pos_xy, cube_size_wh, arm_base_pos_xy, arm_base_size_wh) or \
               check_aabb_overlap(cube_pos_xy, cube_size_wh, tray_pos_xy, tray_size_wh):
                # Cube overlaps, so we need to restart the whole process from tray placement
                tray_placed = False
                break # Break inner loop, outer loop will re-evaluate tray_placed
            else:
                cube_placed = True
                qpos_cube_rot = np.zeros(4)
                mujoco.mju_axisAngle2Quat(qpos_cube_rot, np.array([0, 0, 1]), sigma)
                yellow_cube_qpos = np.array([s_x, s_y, s_z, qpos_cube_rot[0], qpos_cube_rot[1], qpos_cube_rot[2], qpos_cube_rot[3]])

        # If inner loop broke because cube couldn't be placed, restart outer loop
        if not cube_placed:
            continue

        # If we reached here, both tray and cube are placed without overlap
        tray_placed = True # Confirm tray and cube are successfully placed

        # Update environment's qpos for the objects
        qpos_copy = env.data.qpos.copy()
        # Free joint qpos is 7 values (x,y,z,qw,qx,qy,qz)
        if blue_adr != -1: # Ensure joint exists
            qpos_copy[blue_adr : blue_adr + 7] = blue_tray_qpos
        if yellow_adr != -1: # Ensure joint exists
            qpos_copy[yellow_adr : yellow_adr + 7] = yellow_cube_qpos
        env.set_state(qpos_copy, env.data.qvel.copy())


    # 6. set the side_camera position
    # C_z=0.4, distance 2d from arm, on half circle x>0, oriented towards (A_x,A_y)
    cam_Cz = 0.4
    cam_dist = 2 * d # distance from arm (A_x, A_y)

    # Generate a random angle for the half-circle x > 0 (from -pi/2 to pi/2)
    camera_angle = np.random.uniform(-np.pi / 2, np.pi / 2)
    new_cam_pos_x = cam_dist * np.cos(camera_angle)
    new_cam_pos_y = cam_dist * np.sin(camera_angle)
    new_cam_pos = np.array([new_cam_pos_x, new_cam_pos_y, cam_Cz])

    # Orient towards position (A_x, A_y, 0),
    target_pos = np.array([A_x, A_y, 0]) # (0, 0, 0)

    new_cam_quat = get_cam_orientation(new_cam_pos, target_pos)

    # Apply to model if side_camera ID exists
    if side_camera_id != -1:
        env.model.cam_pos[side_camera_id] = new_cam_pos
        env.model.cam_quat[side_camera_id] = new_cam_quat

    print(f"Randomized Yellow Cube Position (x, y, z): ({yellow_cube_qpos[0]:.3f}, {yellow_cube_qpos[1]:.3f}, {yellow_cube_qpos[2]:.3f})")
    print(f"Randomized Yellow Cube Angle (rad vs x-axis): {sigma:.3f}")
    print(f"Randomized Blue Tray Position (x, y, z): ({blue_tray_qpos[0]:.3f}, {blue_tray_qpos[1]:.3f}, {blue_tray_qpos[2]:.3f})")
    print(f"Randomized Blue Tray Angle (rad vs x-axis): {Sigma:.3f}")

    if side_camera_id != -1:
        print(f"Randomized Side Camera Position (x, y, z): ({new_cam_pos[0]:.3f}, {new_cam_pos[1]:.3f}, {new_cam_pos[2]:.3f})")
        print(f"Randomized Side Camera Quat (w,x,y,z): ({new_cam_quat[0]:.3f}, {new_cam_quat[1]:.3f}, {new_cam_quat[2]:.3f}, {new_cam_quat[3]:.3f})")
        

def main():
    parser = argparse.ArgumentParser(description="Set up a MuJoCo environment and save a snapshot image.")
    parser.add_argument(
        "--media-folder",
        type=Path,
        default="../media",
        help="Path to the folder to save the image.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the MuJoCo XML model file.",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="sim_output.png",
        help="Name of the output image file.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        default=False,
        help="If set, randomize object positions (default: False).",
    )
    args = parser.parse_args()

    print(f"MuJoCo version: {mujoco.__version__}")

    model_path = args.model_path.resolve()

    # Create the custom Gymnasium environment
    env = SO101Env(model_path=str(model_path), render_mode="rgb_array", camera_name="side_camera")

    side_camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side_camera")
    top_camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera")
    yellow_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "yellow_cube_to_world")
    blue_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "blue_tray_to_world")

    # Run the simulation, starting from the specified neutral action
    observation, info = env.reset()

    yellow_adr = -1
    blue_adr = -1

    if yellow_joint_id != -1:
        yellow_adr = env.model.jnt_qposadr[yellow_joint_id]

    if blue_joint_id != -1:
        blue_adr = env.model.jnt_qposadr[blue_joint_id]

    if args.random:
        randomize(env, yellow_adr, blue_adr, side_camera_id, top_camera_id)

    # Forward kinematics to update body positions
    mujoco.mj_forward(env.model, env.data)

    # Hold a neutral position
    neutral_action = np.array([ 0.03755415, -1.7234037, 1.6718199, 1.2405578, -1.411793, 0.02459861])

    # Set the initial state of the robot to the neutral action pose
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()
    qpos[:len(neutral_action)] = neutral_action
    qvel[:len(neutral_action)] = 0
    env.set_state(qpos, qvel)

    # Hold the neutral position for a moment to stabilize before testing
    for _ in range(100):
        env.step(neutral_action)

    print("Generating start images...")
    frame = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=side_camera_id)
    image = Image.fromarray(frame)
    output_path = args.media_folder / ("side_" + args.image_name)
    args.media_folder.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    frame = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=top_camera_id)
    image = Image.fromarray(frame)
    output_path = args.media_folder / ("top_" + args.image_name)
    args.media_folder.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Start images saved to media folder.")
    env.close()


if __name__ == "__main__":
    main()
