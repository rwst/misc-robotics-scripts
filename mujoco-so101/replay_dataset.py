import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import mujoco
import datasets
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.wrappers import RecordVideo
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
        # The observation space consists of joint positions and velocities
        # This needs to be flexible based on the model.
        model = mujoco.MjModel.from_xml_path(model_path)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(model.nq + model.nv,), dtype=np.float64
        )

        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            **kwargs,
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


def find_grasp_timestep(episode, gripper_joint_index):
    """
    Finds the timestep of the grasp event in an episode.
    """
    qpos = np.array(episode["observation.state"])
    gripper_qpos = qpos[:, gripper_joint_index]
    gripper_vel = np.diff(gripper_qpos, prepend=gripper_qpos[0])

    peak_closing_vel_idx = np.argmin(gripper_vel)

    for i in range(peak_closing_vel_idx + 1, len(gripper_vel)):
        if np.isclose(gripper_vel[i], 0, atol=1e-3):
            return i

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Replay an episode from a dataset in Mujoco and record a video."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID of the dataset to use.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="The episode index to process from the dataset.",
    )
    parser.add_argument(
        "--robot-xml-file",
        type=str,
        default="so101-assets/so101_new_calib.xml",
        help="Path to the MuJoCo XML file for the robot model for FK.",
    )
    parser.add_argument(
        "--env-xml-file",
        type=str,
        default="so101-assets/so101_with_objects.xml",
        help="Path to the MuJoCo XML file for the environment with objects.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="object_to_grasp",
        help="Name of the object's body/joint in the XML file.",
    )
    parser.add_argument(
        "--video-folder",
        type=Path,
        default="../media",
        help="Path to the folder to save the video.",
    )
    parser.add_argument(
        "--start-image-only",
        action="store_true",
        help="Instead of writing a video, only write a snapshot image of the initial scene.",
    )
    args = parser.parse_args()

    # 1. Load the Dataset
    try:
        print(f"Loading episode {args.episode_index} from dataset '{args.repo_id}'...")
        dataset = datasets.load_dataset(args.repo_id, split="train", streaming=True)
        episode_dataset = dataset.filter(
            lambda example: example["episode_index"] == args.episode_index
        )
        episode_steps = list(episode_dataset)

        if not episode_steps:
            print(f"Episode {args.episode_index} not found or is empty.")
            return

        episode = {
            "observation.state": np.array(
                [step["observation.state"] for step in episode_steps]
            ),
            "action": np.array([step["action"] for step in episode_steps]),
        }
        print("Episode loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset '{args.repo_id}': {e}")
        return

    # 2. Find the Grasp Pose using a temporary model
    try:
        robot_model = mujoco.MjModel.from_xml_path(args.robot_xml_file)
        robot_data = mujoco.MjData(robot_model)
    except Exception as e:
        print(f"Error loading MuJoCo robot model from {args.robot_xml_file}: {e}")
        return

    gripper_joint_index = -1
    grasp_timestep = find_grasp_timestep(episode, gripper_joint_index)

    if grasp_timestep is None:
        print("Could not identify a clear grasp event in the episode.")
        return

    print(f"Grasp event identified at timestep: {grasp_timestep}")

    grasp_qpos = episode["observation.state"][grasp_timestep]
    qpos_radians = np.deg2rad(grasp_qpos)
    robot_data.qpos[: len(qpos_radians)] = qpos_radians
    mujoco.mj_forward(robot_model, robot_data)

    try:
        gripper_site_name = "gripperframe"
        gripper_position = robot_data.site(gripper_site_name).xpos.copy()
        #gripper_position = gripper_position - 0.05
        gripper_orientation_mat = robot_data.site(gripper_site_name).xmat.reshape(3, 3).copy()
        gripper_orientation_quat = np.empty(4)
        mujoco.mju_mat2Quat(gripper_orientation_quat, gripper_orientation_mat.flatten())
        print(f"Estimated grasped object position: {gripper_position}")
    except KeyError:
        print(f"Error: Site '{gripper_site_name}' not found in the robot XML.")
        return

    # 3. Create the final environment
    env = SO101Env(
        model_path=os.path.abspath(args.env_xml_file),
        render_mode="rgb_array",
        camera_name="front_camera",
    )

    # 4. Get object joint info before wrapping (we'll place it after reset)
    object_jnt_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, args.object_name
    )
    qpos_addr = -1
    if object_jnt_id != -1:
        qpos_addr = env.model.jnt_qposadr[object_jnt_id]

    # Move the camera 30cm to the left
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
    if camera_id != -1:
        env.model.cam_pos[camera_id][0] -= 0.3

    if args.start_image_only:
        # Set initial robot state from episode FIRST
        initial_robot_qpos = np.deg2rad(episode["observation.state"][0])
        env.data.qpos[: len(initial_robot_qpos)] = initial_robot_qpos
        env.data.qvel[:] = 0  # Zero out all velocities

        # THEN place object (after setting robot state to avoid overwriting)
        if qpos_addr != -1:
            # Use gripper x,y but correct z-height for the object (0.025m for object1)
            object_position = gripper_position.copy()
            object_position[2] = 0.025  # Object z-height above ground
            env.data.qpos[qpos_addr : qpos_addr + 3] = object_position
            env.data.qpos[qpos_addr + 3 : qpos_addr + 7] = gripper_orientation_quat
            print(f"Placed object '{args.object_name}' at position: ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
            print(f"  (original gripper z was: {gripper_position[2]:.3f})")
            print(f"  qpos_addr: {qpos_addr}, total qpos size: {env.data.qpos.size}")
        else:
            print(f"WARNING: Object '{args.object_name}' not found in XML!")

        # Update physics to reflect the new state
        mujoco.mj_forward(env.model, env.data)

        print("Generating start image only...")
        frame = env.render()
        image = Image.fromarray(frame)
        output_path = (
            args.video_folder
            / f"replay_{args.repo_id.replace('/', '_')}_ep{args.episode_index}_start.png"
        )
        args.video_folder.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Start image saved to {output_path}")
        env.close()
        return

    # 6. Wrap for video recording and replay actions
    video_name_prefix = (
        f"replay_{args.repo_id.replace('/', '_')}_ep{args.episode_index}"
    )
    env = RecordVideo(env, str(args.video_folder), name_prefix=video_name_prefix)

    env.reset()

    # Place object after reset (minimal approach - just set position)
    if qpos_addr != -1:
        # Use gripper x,y but correct z-height for the object (0.025m for object1)
        object_position = gripper_position.copy()
        object_position[2] = 0.025  # Object z-height above ground
        env.unwrapped.data.qpos[qpos_addr : qpos_addr + 3] = object_position
        env.unwrapped.data.qpos[qpos_addr + 3 : qpos_addr + 7] = gripper_orientation_quat
        print(f"Placed object '{args.object_name}' at grasp location.")
        print(f"  Position (x, y, z): ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
        print(f"  (original gripper z was: {gripper_position[2]:.3f})")

    actions = episode["action"]

    max_steps_per_move = 500
    movement_epsilon = 1e-3  # Stop if position change norm is less than this
    num_joints = env.action_space.shape[0]

    for i, action in enumerate(actions):
        print(f"Executing action {i+1}/{len(actions)}...")
        scaled_action = action / 100 * np.where(action > 0, env.action_space.high, -env.action_space.low)

        previous_pos = np.full(num_joints, np.inf)
        for step in range(max_steps_per_move):
            observation, reward, terminated, truncated, info = env.step(scaled_action)
            current_pos = observation[:num_joints]

            # Check if the movement has stopped
            norm = np.linalg.norm(current_pos - previous_pos)
            if norm < movement_epsilon:
                print(f"  Movement stabilized in {step + 1} steps.")
                break

            previous_pos = current_pos

            if terminated or truncated:
                print("  Episode terminated or truncated during action execution.")
                break
        else:
            # This block executes if the for loop completes without a 'break'
            print(f"  Warning: Action timed out after {max_steps_per_move} steps. Norm = {norm}")

        if terminated or truncated:
            print("Replay finished due to episode termination.")
            break

    env.close()

    print(f"Simulation finished. Video saved in the '{args.video_folder}' directory with prefix '{video_name_prefix}'.")


if __name__ == "__main__":
    main()
