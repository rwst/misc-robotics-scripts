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

def set_camera_orientations(env, camera_ids, target_pos=None):
    """
    Sets the orientation of specified cameras to look towards a target position.

    :param env: The MuJoCo environment.
    :param camera_ids: A list of camera IDs to orient.
    :param target_pos: The position to look at. If None, defaults to the origin (0,0,0).
    """
    if target_pos is None:
        target_pos = np.array([0.0, 0.0, 0.0])

    for cam_id in camera_ids:
        if cam_id != -1:
            cam_pos = env.model.cam_pos[cam_id]
            cam_quat = get_cam_orientation(cam_pos, target_pos)
            env.model.cam_quat[cam_id] = cam_quat
            
            cam_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            print(f"Oriented '{cam_name}' to look at target {target_pos}.")
            print(f"  - Position: {np.round(cam_pos, 3)}")
            print(f"  - New Quaternion: {np.round(cam_quat, 3)}")


def randomize(env, object1_adr, object2_adr, side_camera_id, top_camera_id):
    """
    Randomizes the positions and orientations of object1 and object2
    within a defined sandbox, checking for overlaps.
    """
    # Constants from the problem description
    d = 0.4
    A_w, A_h = 0.05, 0.05  # Arm base dimensions (full width, height)
    s_w, s_h = 0.02, 0.01  # object1 dimensions (full width, height)
    S_w, S_h = 0.1, 0.15  # object2 dimensions (full width, height)

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
    S_z = 0.01  # object2 Z position
    s_z = 0.025 # object1 Z position

    object2_placed = False
    while not object2_placed:
        # 2. compute random object2 position (S_x,S_y) within sandbox with random orientation Sigma
        # Calculate bounds for object2 center to ensure its center is within sandbox
        object2_min_center_x = O_x
        object2_max_center_x = O_x + O_w
        object2_min_center_y = O_y
        object2_max_center_y = O_y + O_h

        S_x = np.random.uniform(object2_min_center_x, object2_max_center_x)
        S_y = np.random.uniform(object2_min_center_y, object2_max_center_y)
        Sigma = np.random.uniform(-np.pi, np.pi)  # Random orientation (radians)

        object2_pos_xy = np.array([S_x, S_y])
        object2_size_wh = np.array([S_w, S_h]) # Using nominal size for AABB overlap check

        # 3. if object2 overlaps arm base, reset and go to 1
        if check_aabb_overlap(object2_pos_xy, object2_size_wh, arm_base_pos_xy, arm_base_size_wh):
            continue # Retry object2 placement

        # Store tentative object2 placement data (x, y, z, qw, qx, qy, qz)
        qpos_object2_rot = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qpos_object2_rot, np.array([0, 0, 1]), Sigma) # Rotate around Z-axis
        object2_qpos = np.array([S_x, S_y, S_z, qpos_object2_rot[0], qpos_object2_rot[1], qpos_object2_rot[2], qpos_object2_rot[3]])

        object1_placed = False
        while not object1_placed:
            # 4. compute random object1 position (s_x,s_y) within sandbox with random orientation sigma
            object1_min_center_x = O_x
            object1_max_center_x = O_x + O_w
            object1_min_center_y = O_y
            object1_max_center_y = O_y + O_h

            s_x = np.random.uniform(object1_min_center_x, object1_max_center_x)
            s_y = np.random.uniform(object1_min_center_y, object1_max_center_y)
            sigma = np.random.uniform(-np.pi, np.pi)  # Random orientation (radians)

            object1_pos_xy = np.array([s_x, s_y])
            object1_size_wh = np.array([s_w, s_h]) # Using nominal size for AABB overlap check

            # 5. if object1 overlaps object2 or arm base, go to 3 (reset object2)
            if check_aabb_overlap(object1_pos_xy, object1_size_wh, arm_base_pos_xy, arm_base_size_wh) or \
               check_aabb_overlap(object1_pos_xy, object1_size_wh, object2_pos_xy, object2_size_wh):
                # object1 overlaps, so we need to restart the whole process from object2 placement
                object2_placed = False
                break # Break inner loop, outer loop will re-evaluate object2_placed
            else:
                object1_placed = True
                qpos_object1_rot = np.zeros(4)
                mujoco.mju_axisAngle2Quat(qpos_object1_rot, np.array([0, 0, 1]), sigma)
                object1_qpos = np.array([s_x, s_y, s_z, qpos_object1_rot[0], qpos_object1_rot[1], qpos_object1_rot[2], qpos_object1_rot[3]])

        # If inner loop broke because object1 couldn't be placed, restart outer loop
        if not object1_placed:
            continue

        # If we reached here, both object2 and object1 are placed without overlap
        object2_placed = True # Confirm object2 and object1 are successfully placed

        # Update environment's qpos for the objects
        qpos_copy = env.data.qpos.copy()
        # Free joint qpos is 7 values (x,y,z,qw,qx,qy,qz)
        if object2_adr != -1: # Ensure joint exists
            qpos_copy[object2_adr : object2_adr + 7] = object2_qpos
        if object1_adr != -1: # Ensure joint exists
            qpos_copy[object1_adr : object1_adr + 7] = object1_qpos
        env.set_state(qpos_copy, env.data.qvel.copy())


    print(f"Randomized object1 Position (x, y, z): ({object1_qpos[0]:.3f}, {object1_qpos[1]:.3f}, {object1_qpos[2]:.3f})")
    print(f"Randomized object1 Angle (rad vs x-axis): {sigma:.3f}")
    print(f"Randomized object2 Position (x, y, z): ({object2_qpos[0]:.3f}, {object2_qpos[1]:.3f}, {object2_qpos[2]:.3f})")
    print(f"Randomized object2 Angle (rad vs x-axis): {Sigma:.3f}")
        

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
    env = SO101Env(model_path=str(model_path), render_mode="rgb_array", camera_name="side_camera", width=640, height=480)

    side_camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side_camera")
    top_camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera")
    object1_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "object1_to_world")
    object2_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "object2_to_world")

    # Orient cameras towards the robot base
    set_camera_orientations(env, [side_camera_id, top_camera_id])

    # Run the simulation, starting from the specified neutral action
    observation, info = env.reset()

    object1_adr = -1
    object2_adr = -1

    if object1_joint_id != -1:
        object1_adr = env.model.jnt_qposadr[object1_joint_id]

    if object2_joint_id != -1:
        object2_adr = env.model.jnt_qposadr[object2_joint_id]

    if args.random:
        randomize(env, object1_adr, object2_adr, side_camera_id, top_camera_id)

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
