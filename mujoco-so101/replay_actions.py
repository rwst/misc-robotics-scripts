import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.wrappers import RecordVideo


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
        # Assume action is 6D positions; extend for full qpos if needed
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(6)]
        self.data.qpos[self.model.jnt_qposadr[joint_ids]] = action
        # Optional: Set velocities to zero or interpolate (e.g., diff from prev)
        self.data.qvel[:6] = 0.0  # Or compute: (action - prev_action) / timestep
        mujoco.mj_forward(self.model, self.data)  # Update kinematics (positions/orientations)
        # Advance time manually if needed: self.simulate(self.timestep)
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
        default="episode_0_states.npy",
        help="Path to the .npy file containing the actions.",
    )
    parser.add_argument(
        "--video-folder",
        type=Path,
        default="../media",
        help="Path to the folder to save the video.",
    )
    args = parser.parse_args()

    # Get the absolute path to the XML file
    model_path = os.path.join(os.path.dirname(__file__), "../so101-assets/so101_new_calib.xml")

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

    # Wrap the environment to record a video
    video_name_prefix = f"replay_{args.actions_path.stem}"
    env = RecordVideo(env, str(args.video_folder), name_prefix=video_name_prefix)

    # Run the simulation
    observation, info = env.reset()
    print(env.action_space)
    n = 0
    for action in actions:
        n = n + 1
        print("{} {}".format(n, action))
        action = action / 100 * np.where(action > 0, env.action_space.high, -env.action_space.low)
        print("{} {}".format(n, action))
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode terminated or truncated. Resetting environment.")
            observation, info = env.reset()

    env.close()

    print(f"Simulation finished. Video saved in the '{args.video_folder}' directory with prefix '{video_name_prefix}'.")


if __name__ == "__main__":
    main()
