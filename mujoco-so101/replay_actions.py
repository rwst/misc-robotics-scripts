import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import mujoco
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
        # 6 joints -> 6 qpos + 6 qvel = 12 dimensions
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)

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


def main():
    parser = argparse.ArgumentParser(description="Replay actions from a .npy file in Mujoco and record a video.")
    parser.add_argument(
        "--actions-path",
        type=Path,
        default="episode_0_actions.npy",
        help="Path to the .npy file containing the actions.",
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

    # Get the absolute path to the XML file
    model_path = os.path.join(os.path.dirname(__file__), "so101-assets/so101_new_calib.xml")

    # Load actions
    print(f"Loading actions from {args.actions_path}")
    try:
        actions = np.load(args.actions_path)
    except FileNotFoundError:
        print(f"Error: Actions file not found at {args.actions_path}")
        return
    print(f"Loaded {len(actions)} actions.")

    # Create the custom Gymnasium environment
    env = SO101Env(model_path=model_path, render_mode="rgb_array", camera_name="front_camera")

    # Move the camera 30cm to the left
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
    if camera_id != -1:
        print("Moving camera to the left...")
        # In MuJoCo's camera frame, +Y is typically to the left.
        env.model.cam_pos[camera_id][0] -= 0.3
    else:
        print("Warning: Could not find camera named 'front_camera'.")

    # Run the simulation, starting from the specified neutral action
    observation, info = env.reset()

    # Hold a neutral position
    neutral_action = np.array([ 0.03755415, -1.7234037, 1.6718199, 1.2405578, -1.411793, 0.02459861])

    # Set the initial state of the robot to the neutral action pose
    neutral_qvel = np.zeros_like(neutral_action)
    env.set_state(neutral_action, neutral_qvel)

    # Hold the neutral position for a moment to stabilize before testing
    for _ in range(100):
        env.step(neutral_action)

    if args.start_image_only:
        print("Generating start image only...")
        frame = env.render()
        image = Image.fromarray(frame)
        output_path = args.video_folder / f"replay_{args.actions_path.stem}_start.png"
        args.video_folder.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Start image saved to {output_path}")
        env.close()
        return

    # Wrap the environment to record a video
    video_name_prefix = f"replay_{args.actions_path.stem}"
    env = RecordVideo(env, str(args.video_folder), name_prefix=video_name_prefix)

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
