import os, torch
import numpy as np
import mujoco
from dataclasses import dataclass
from lerobot.datasets.utils import flatten_dict
from lerobot.envs.configs import EnvConfig
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from gymnasium.spaces import Box, Dict, Space
from gymnasium.envs.mujoco import MujocoEnv

# --- Configuration ---
# TODO: Replace with your specific Hugging Face repository IDs.
POLICY_REPO_ID = "lerobot/smolvla_base"
# TODO: Replace with the instruction for the task.
INSTRUCTION = "put the yellow brick onto the blue tray"
# Path to the MuJoCo XML file for the environment.
MODEL_XML_PATH = "mujoco-so101/so101-assets/so101_new_calib.xml"
# Name of the camera to use for image observations.
CAMERA_NAME = "camera1"
# Image size for the policy observation.
IMG_WIDTH = 256
IMG_HEIGHT = 256


@dataclass
class SO101EnvConfig(EnvConfig):
    """A concrete EnvConfig for the custom SO101 environment."""
    name: str = "SO101"
    observation_space: Space = None
    action_space: Space = None
    control_freq: int = -1

    @property
    def gym_kwargs(self) -> dict[str, any]:
        return {}


class SO101Env(MujocoEnv):
    """
    Custom MuJoCo environment for the SO101 robot that provides observations
    compatible with the SmolVLA model (image and state).
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 500,
    }

    def __init__(self, model_path, frame_skip=1, **kwargs):
        # Define the observation space to match SmolVLA's expected inputs.
        # It's a dictionary with 'images' and 'state'.
        observation_space = Dict({
            "images": Dict({
                CAMERA_NAME: Box(
                    low=0, high=255, shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8
                )
            }),
            "state": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
        })

        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            width=IMG_WIDTH,
            height=IMG_HEIGHT,
            **kwargs,
        )

    def _get_obs(self):
        """
        Returns the observation from the environment in the format expected by SmolVLA.
        """
        # Get the state vector (joint positions and velocities).
        state = np.concatenate([self.data.qpos, self.data.qvel]).ravel()
        # Get the image observation.
        image = self.render()
        return {"images": {CAMERA_NAME: image}, "state": state}

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
    """
    This script demonstrates how to use a pre-trained SmolVLA policy and its
    associated processors to run inference in a custom MuJoCo environment.
    """
    print(f"Loading policy from: {POLICY_REPO_ID}")
    device = "cpu"

    # 1. Initialize the custom MuJoCo environment to get its spaces.
    # Construct the absolute path to the model XML file.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_XML_PATH)
    print(f"Initializing custom MuJoCo environment from: {model_path}")
    env = SO101Env(model_path=str(model_path), render_mode="rgb_array", camera_name=CAMERA_NAME)

    # 2. Create an EnvConfig from the environment instance.
    env_cfg = SO101EnvConfig(
        observation_space=env.observation_space,
        action_space=env.action_space,
        control_freq=env.metadata["render_fps"],
    )

    # 3. Load the policy configuration from the Hugging Face Hub.
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_REPO_ID)
    policy_cfg.pretrained_path = POLICY_REPO_ID
    policy_cfg.device = device

    # 4. Load the policy and processors.
    policy = make_policy(policy_cfg, env_cfg=env_cfg)
    pre_processor, post_processor = make_smolvla_pre_post_processors(config=policy_cfg)
    policy.eval()

    # 5. Run the inference loop.
    obs, info = env.reset()
    terminated = truncated = False

    print("Starting inference loop. Press Ctrl+C to exit.")
    try:
        while not terminated and not truncated:
            # a. Prepare the input for the pre-processor.
            # The pre-processor's _forward method expects a dictionary that looks like a Transition object.
            input_data = {
                "observation": obs,
                "action": None,
                "complementary_data": {"task": INSTRUCTION},
            }

            # b. Pre-process the observation, bypassing the batch converter.
            processed_batch = pre_processor._forward(input_data)

            # Manually convert to tensor, add a batch dimension, and permute channels for the image.
            # The policy expects (b, c, h, w), but the output is (h, w, c).
            img_array = processed_batch["observation"]["images"]["camera1"]
            # Copy the array to resolve negative stride issues from the renderer.
            img_tensor = torch.from_numpy(img_array.copy())
            processed_batch["observation"]["images"]["camera1"] = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            # Also convert the state to a batched tensor.
            state_array = processed_batch["observation"]["state"]
            processed_batch["observation"]["state"] = torch.from_numpy(state_array).unsqueeze(0).float()

            print(processed_batch)

            # c. Get the action from the policy.
            with torch.no_grad():
                flat_batch = flatten_dict(processed_batch)
                # The policy expects dot-separated keys, but flatten_dict uses slashes.
                flat_batch = {k.replace("/", "."): v for k, v in flat_batch.items()}
                action_prediction = policy.select_action(flat_batch)

            # d. Post-process the action.
            action_processed = post_processor({"action": action_prediction})
            action_env = action_processed["action"]

            # e. Step the environment.
            obs, reward, terminated, truncated, info = env.step(action_env.cpu().numpy())
            
            # The environment is rendered via the window created with `render_mode="human"`.
            # No explicit `env.render()` call is needed in the loop.

    except KeyboardInterrupt:
        print("\nInference loop interrupted by user.")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
