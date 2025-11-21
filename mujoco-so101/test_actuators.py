import argparse
import os
from pathlib import Path
import time

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
        self.data.ctrl[:action.shape[0]] = action
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        observation = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser(description="Test actuators by moving them sequentially and record a video.")
    parser.add_argument(
        "--video-folder",
        type=Path,
        default="../media",
        help="Path to the folder to save the video.",
    )
    parser.add_argument(
        "--env-xml-file",
        type=str,
        default="so101-assets/so101_new_calib.xml",
        help="Path to the MuJoCo XML model file (relative to script directory or absolute).",
    )
    args = parser.parse_args()

    # Get the absolute path to the XML file
    model_path = os.path.join(os.path.dirname(__file__), args.env_xml_file)

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

    # Wrap the environment to record a video
    video_name_prefix = "test_actuators"
    env = RecordVideo(env, str(args.video_folder), name_prefix=video_name_prefix)

    # Run the simulation, starting from the specified neutral action
    observation, info = env.reset()
    
    num_joints = env.action_space.shape[0]
    
    # Hold a neutral position for the robot
    neutral_action = np.array([ 0.03755415, -1.7234037, 1.6718199, 1.2405578, -1.411793, 0.02459861])

    # Get full state from environment initialization (includes objects if present)
    full_qpos = env.unwrapped.init_qpos.copy()
    full_qvel = env.unwrapped.init_qvel.copy()

    # Set robot joint positions to neutral (first num_joints DoF)
    full_qpos[:num_joints] = neutral_action
    full_qvel[:num_joints] = 0  # Zero velocity for robot joints

    # Set the complete state (works for models with or without objects)
    env.unwrapped.set_state(full_qpos, full_qvel)
    
    # Hold the neutral position for a moment to stabilize before testing
    for _ in range(100):
        env.step(neutral_action)

    max_steps_per_move = 2000
    tolerance = 0.05  # Radians (about 3 degrees)
    movement_epsilon = 1e-6  # Stop if position changes by less than this

    # Test each joint sequentially
    for i in range(num_joints):
        # Ensure the actuator's target is a joint before proceeding
        if env.unwrapped.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            # Get the ID (the first element) and name of the joint controlled by the actuator
            joint_id = env.unwrapped.model.actuator_trnid[i, 0]
            joint_name = mujoco.mj_id2name(env.unwrapped.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            
            # Get the address of the joint's position value in the qpos array
            qpos_addr = env.unwrapped.model.jnt_qposadr[joint_id]
            
            # Get the position limits for the joint
            joint_min_pos = env.unwrapped.model.jnt_range[joint_id][0]
            joint_max_pos = env.unwrapped.model.jnt_range[joint_id][1]

            print(f"\nTesting Actuator {i} (Joint: '{joint_name}')")
            action = np.copy(neutral_action)

            # --- Move to min position ---
            print(f"  Moving to min position ({joint_min_pos:.2f})...")
            action[i] = env.action_space.low[i]
            previous_pos = np.inf
            for step in range(max_steps_per_move):
                obs, _, _, _, _ = env.step(action)
                current_pos = obs[qpos_addr]
                if abs(current_pos - joint_min_pos) < tolerance:
                    print(f"  Reached min in {step + 1} steps.")
                    break
                if abs(current_pos - previous_pos) < movement_epsilon:
                    print(f"  Movement stopped at {current_pos:.2f} after {step + 1} steps.")
                    break
                previous_pos = current_pos
            else:
                print(f"  Warning: Timed out. Final position: {current_pos:.2f}")
            for _ in range(50): env.step(action)

            # --- Move to max position ---
            print(f"  Moving to max position ({joint_max_pos:.2f})...")
            action[i] = env.action_space.high[i]
            previous_pos = -np.inf
            for step in range(max_steps_per_move):
                obs, _, _, _, _ = env.step(action)
                current_pos = obs[qpos_addr]
                if abs(current_pos - joint_max_pos) < tolerance:
                    print(f"  Reached max in {step + 1} steps.")
                    break
                if abs(current_pos - previous_pos) < movement_epsilon:
                    print(f"  Movement stopped at {current_pos:.2f} after {step + 1} steps.")
                    break
                previous_pos = current_pos
            else:
                print(f"  Warning: Timed out. Final position: {current_pos:.2f}")
            for _ in range(50): env.step(action)

            # --- Return to min position ---
            print(f"  Returning to min position ({joint_min_pos:.2f})...")
            action[i] = env.action_space.low[i]
            previous_pos = np.inf
            for step in range(max_steps_per_move):
                obs, _, _, _, _ = env.step(action)
                current_pos = obs[qpos_addr]
                if abs(current_pos - joint_min_pos) < tolerance:
                    print(f"  Reached min in {step + 1} steps.")
                    break
                if abs(current_pos - previous_pos) < movement_epsilon:
                    print(f"  Movement stopped at {current_pos:.2f} after {step + 1} steps.")
                    break
                previous_pos = current_pos
            else:
                print(f"  Warning: Timed out. Final position: {current_pos:.2f}")
            for _ in range(50): env.step(action)
        else:
            print(f"\nSkipping Actuator {i} as it does not target a joint.")

    # Return all joints to neutral
    print("\nReturning all joints to neutral position.")
    for _ in range(200):
        env.step(neutral_action)

    env.close()

    print(f"Simulation finished. Video saved in the '{args.video_folder}' directory with prefix '{video_name_prefix}'.")


if __name__ == "__main__":
    main()
