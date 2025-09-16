import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
import numpy as np
import os


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
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
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
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


# Get the absolute path to the XML file
model_path = os.path.join(os.path.dirname(__file__), "../so101-assets/so101_new_calib.xml")

# Create the custom Gymnasium environment
env = SO101Env(model_path=model_path, render_mode="rgb_array", camera_name="front_camera")

# Wrap the environment to record a video
video_folder = "../media"
env = RecordVideo(env, video_folder, name_prefix="so101_manual")

# Run the simulation
observation, info = env.reset()
for i in range(400):
    # Use a simple sinusoidal motion for smoother movement of all joints
    t = i * 0.05
    action = 0.5 * np.sin(t) * np.ones(env.action_space.shape)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

print(f"Simulation finished. Video saved in the '{video_folder}' directory.")
