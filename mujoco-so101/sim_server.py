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
    args = parser.parse_args()

    model_path = args.model_path.resolve()

    # Create the custom Gymnasium environment
    env = SO101Env(model_path=str(model_path), render_mode="rgb_array", camera_name="front_camera")

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

    print("Generating start image...")
    frame = env.render()
    image = Image.fromarray(frame)
    output_path = args.media_folder / "sim_output.png"
    args.media_folder.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Start image saved to {output_path}")
    env.close()


if __name__ == "__main__":
    main()
