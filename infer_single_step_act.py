import time
from draccus import field, wrap
from typing import Dict, Any
from dataclasses import dataclass

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import EnvConfig

import torch
from gymnasium.spaces import Box, Space

"""
This script needs a version of lerobot where the pull request
https://github.com/huggingface/lerobot/pull/1771
is merged, or the issue is fixed by similar means.
"""


@dataclass
class DummyEnvConfig(EnvConfig):
    """A concrete EnvConfig for dummy environments that does not require gym registration."""
    name: str = "dummy"
    observation_space: Space = None
    action_space: Space = None
    control_freq: int = -1

    @property
    def gym_kwargs(self) -> dict[str, any]:
        return {}

@dataclass
class InferSingleStepConfig:
    """Configuration for single-step inference."""

    policy_path: str = field(
        default="",
        metadata={"help": "Path or Hugging Face repo ID of the trained policy."},
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run the policy on (e.g., 'cuda', 'cpu')."},
    )
    # NOTE: These dummy shapes must match the policy's expected input.
    # Adjust these if your policy uses different input dimensions (e.g., multiple cameras).
    dummy_image_shape: tuple[int, int, int] = field(
        default=(3, 128, 128),
        metadata={"help": "Shape of the dummy image observation (C, H, W)."},
    )
    dummy_state_dim: int = field(
        default=7,
        metadata={"help": "Dimension of the dummy state observation."},
    )

def get_dummy_env_config(cfg: InferSingleStepConfig, policy_cfg: PreTrainedConfig) -> EnvConfig:
    """
    Creates a minimal EnvConfig required by make_policy when loading a pretrained model.
    
    This is necessary because make_policy requires either ds_meta or env_cfg to derive 
    feature shapes, even when loading a pretrained model whose config should already 
    contain them.
    """
    # Determine action dimension. If output_features is defined in the loaded config, use it.
    # Otherwise, fall back to using dummy_state_dim as a guess (common for 7-DoF robots).
    action_dim = cfg.dummy_state_dim
    if policy_cfg.output_features:
        # Assuming the primary action feature is a 1D vector
        for ft in policy_cfg.output_features.values():
            if hasattr(ft, 'shape') and ft.shape:
                action_dim = ft.shape[-1]
                break

    # Define observation space (using Box from gym.spaces)
    # State space: (D,)
    state_space = Box(low=-float('inf'), high=float('inf'), shape=(cfg.dummy_state_dim,), dtype=float)
    # Image space: (C, H, W)
    image_space = Box(low=0, high=255, shape=cfg.dummy_image_shape, dtype=float)

    observation_space = {"image": image_space, "state": state_space}
    
    # Action space: (A,)
    action_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=float)

    return DummyEnvConfig(observation_space=observation_space, action_space=action_space)

@wrap()
def infer_single_step(cfg: InferSingleStepConfig):
    """
    Loads a policy and performs a single-step inference using dummy input data.
    """
    print(f"Loading policy from: {cfg.policy_path}")
    device = cfg.device

    # 1. Load the policy configuration
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)

    # Attempt to derive state dimension from the policy's input features
    if policy_cfg.input_features and "observation.state" in policy_cfg.input_features:
        state_feature = policy_cfg.input_features["observation.state"]
        if hasattr(state_feature, 'shape') and state_feature.shape:
            cfg.dummy_state_dim = state_feature.shape[-1]
            print(f"Inferred state dimension from policy config: {cfg.dummy_state_dim}")
    
    # 2. Prepare configuration for loading
    policy_cfg.pretrained_path = cfg.policy_path
    policy_cfg.device = device

    # 3. Create a dummy environment configuration to satisfy make_policy requirements
    dummy_env_cfg = get_dummy_env_config(cfg, policy_cfg)

    # 4. Load the policy model using the factory
    policy: PreTrainedPolicy = make_policy(policy_cfg, env_cfg=dummy_env_cfg)
    policy.eval()

    # 2. Create dummy input observations
    # We assume a batch size of 1 for single-step inference.
    # The input must be a dictionary matching the structure expected by the policy.

    # Dummy image (e.g., 1x3x128x128)
    dummy_image = torch.rand(
        (1, *cfg.dummy_image_shape),
        dtype=torch.float32,
        device=device
    )

    # Dummy state (e.g., 1x7)
    dummy_state = torch.rand(
        (1, cfg.dummy_state_dim),
        dtype=torch.float32,
        device=device
    )

    # Construct the observation dictionary
    dummy_obs: Dict[str, Any] = {
        "observation.images.front": dummy_image,
        "observation.images.gripperR": dummy_image,
        "observation.state": dummy_state,
    }

    # ACT policies require the future action window (horizon) to be specified.
    # We check if the policy has a horizon attribute (common for ACT/Diffusion).
    if hasattr(policy, "horizon"):
        # ACT/Diffusion policies often require a sequence of observations for context.
        # We replicate the single observation to simulate a sequence of length 1.
        # Shape: (B, T, ...) -> (1, 1, ...)
        for key in dummy_obs:
            dummy_obs[key] = dummy_obs[key].unsqueeze(1)

        # ACT/Diffusion also requires the action sequence length (horizon)
        # We pass the horizon as part of the context for prediction.
        dummy_obs["context"] = {"horizon": torch.tensor([policy.horizon], device=device)}

    print(f"Running inference with dummy input shapes:")
    for k, v in dummy_obs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")

    # 3. Run inference
    with torch.no_grad():
        start_time = time.time()
        action_prediction = policy.select_action(dummy_obs)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000

    # 4. Display results
    print("\n--- Inference Result ---")
    print(f"Inference Time: {inference_time_ms:.2f} ms")
    print(f"Predicted Action Type: {type(action_prediction)}")

    if isinstance(action_prediction, torch.Tensor):
        print(f"Predicted Action Shape: {tuple(action_prediction.shape)}")
        print(f"Predicted Action: {action_prediction.flatten().tolist()}")
    else:
        print(f"Predicted Action Content: {action_prediction}")

    print("------------------------")
    print("Single-step inference complete.")


if __name__ == "__main__":
    infer_single_step()
