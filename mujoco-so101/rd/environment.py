"""
SO101 MuJoCo Environment and setup utilities.
"""

import os
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv


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
        # Validate action dimension
        if action.shape[0] != self.model.nu:
            raise ValueError(
                f"Action dimension mismatch: expected {self.model.nu} actuators, "
                f"got action with shape {action.shape}"
            )

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        observation = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


def create_environment(env_xml_file, camera_name="front_camera", object_name="object_to_grasp"):
    """
    Creates and configures the SO101 MuJoCo environment.

    Args:
        env_xml_file: Path to the environment XML file
        camera_name: Name of the camera to use
        object_name: Name of the object in the scene

    Returns:
        tuple: (env, object_jnt_id, qpos_addr) or (None, -1, -1) if creation fails
    """
    try:
        env = SO101Env(
            model_path=os.path.abspath(env_xml_file),
            render_mode="rgb_array",
            camera_name=camera_name,
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        return None, -1, -1

    # Get object joint info
    object_jnt_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, object_name
    )
    qpos_addr = -1
    if object_jnt_id != -1:
        qpos_addr = env.model.jnt_qposadr[object_jnt_id]

    # Move the camera 30cm to the left
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id != -1:
        env.model.cam_pos[camera_id][0] -= 0.3

    return env, object_jnt_id, qpos_addr


def place_object_in_scene(env_data, qpos_addr, gripper_position, gripper_orientation_quat, object_name):
    """
    Places object in the scene at the specified position.

    Args:
        env_data: MuJoCo data object
        qpos_addr: qpos address for the object joint
        gripper_position: 3D position of gripper (numpy array)
        gripper_orientation_quat: Quaternion orientation (numpy array)
        object_name: Name of the object for logging
    """
    if qpos_addr == -1:
        print(f"WARNING: Object '{object_name}' not found in XML!")
        return

    if gripper_position is None:
        print(f"Object '{object_name}' found but no position available - object not placed")
        return

    object_position = gripper_position.copy()
    object_position[2] = 0.025  # Object z-height above ground
    env_data.qpos[qpos_addr : qpos_addr + 3] = object_position
    env_data.qpos[qpos_addr + 3 : qpos_addr + 7] = gripper_orientation_quat
    print(f"Placed object '{object_name}' at position: ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
    print(f"  (original gripper z was: {gripper_position[2]:.3f})")
    if hasattr(env_data, 'qpos'):
        print(f"  qpos_addr: {qpos_addr}, total qpos size: {env_data.qpos.size}")
