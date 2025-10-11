import time
from draccus import field, wrap
from typing import Dict, Any
from dataclasses import dataclass

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import EnvConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.datasets.utils import flatten_dict

import torch
from gymnasium.spaces import Box, Space


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
        default="lerobot/smolvla_base",
        metadata={"help": "Path or Hugging Face repo ID of the trained policy."},
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run the policy on (e.g., 'cuda', 'cpu')."},
    )
    dummy_image_shape: tuple[int, int, int] = field(
        default=(3, 256, 256),
        metadata={"help": "Shape of the dummy image observation (C, H, W)."},
    )
    dummy_state_dim: int = field(
        default=7,
        metadata={"help": "Dimension of the dummy state observation."},
    )
    instruction: str = field(
        default="put the yellow brick onto the blue tray",
        metadata={"help": "Language instruction for the policy."},
    )


def get_dummy_env_config(cfg: InferSingleStepConfig, policy_cfg: PreTrainedConfig) -> EnvConfig:
    """
    Creates a minimal EnvConfig required by make_policy when loading a pretrained model.
    """
    action_dim = cfg.dummy_state_dim
    if policy_cfg.output_features:
        for ft in policy_cfg.output_features.values():
            if hasattr(ft, 'shape') and ft.shape:
                action_dim = ft.shape[-1]
                break

    state_space = Box(low=-float('inf'), high=float('inf'), shape=(cfg.dummy_state_dim,), dtype=float)
    image_space = Box(low=0, high=255, shape=cfg.dummy_image_shape, dtype=float)
    observation_space = {"image": image_space, "state": state_space}
    action_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=float)
    return DummyEnvConfig(observation_space=observation_space, action_space=action_space)

@wrap()
def infer_single_step(cfg: InferSingleStepConfig):
    """
    Loads a smolvla policy and performs a single-step inference using dummy input data.
    """
    print(f"Loading policy from: {cfg.policy_path}")
    device = cfg.device

    # 1. Load the policy configuration
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)

    if policy_cfg.input_features and "observation.state" in policy_cfg.input_features:
        state_feature = policy_cfg.input_features["observation.state"]
        if hasattr(state_feature, 'shape') and state_feature.shape:
            cfg.dummy_state_dim = state_feature.shape[-1]
            print(f"Inferred state dimension from policy config: {cfg.dummy_state_dim}")
    
    # 2. Prepare configuration for loading
    policy_cfg.pretrained_path = cfg.policy_path
    policy_cfg.device = device

    # 3. Create a dummy environment configuration
    dummy_env_cfg = get_dummy_env_config(cfg, policy_cfg)

    # 4. Load the policy model and create the pre-processor
    policy: PreTrainedPolicy = make_policy(policy_cfg, env_cfg=dummy_env_cfg)
    pre_processor, _ = make_smolvla_pre_post_processors(config=policy_cfg)
    policy.eval()

    # 5. Create dummy input observations
    dummy_image = torch.rand((1, *cfg.dummy_image_shape), dtype=torch.float32, device=device)
    dummy_state = torch.rand((1, cfg.dummy_state_dim), dtype=torch.float32, device=device)

    # Construct the nested observation dictionary with the specific keys expected by the model config.
    observation: Dict[str, Any] = {
        "images": {
            "camera1": dummy_image,
            "camera2": dummy_image,
            "camera3": dummy_image,
        },
        "state": dummy_state,
    }
    
    # The processor expects a dictionary that looks like a Transition object.
    input_data = {
        "observation": observation,
        "action": None,
        "complementary_data": {"task": cfg.instruction},
    }

    print(f"Running inference with dummy input data...")

    # 6. Run inference
    with torch.no_grad():
        start_time = time.time()
        
        # The pre-processor expects a nested dictionary and handles tokenization, etc.
        processed_obs = pre_processor._forward(input_data)

        # The policy's select_action method expects a flat dictionary.
        # We flatten the entire processed output dictionary.
        flat_batch = flatten_dict(processed_obs)

        # The policy expects keys with '.' as a separator, and the processor can sometimes create
        # duplicate prefixes like 'observation.observation.'. We clean this up in one pass.
        flat_batch = {
            k.replace("/", ".").replace("observation.observation.", "observation."): v
            for k, v in flat_batch.items()
        }

        action_prediction = policy.select_action(flat_batch)
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000

    # 7. Display results
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
