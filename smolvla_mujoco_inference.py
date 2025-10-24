import os, torch, argparse
import numpy as np
import mujoco
from pathlib import Path
from dataclasses import dataclass
from lerobot.datasets.utils import flatten_dict
from lerobot.envs.configs import EnvConfig
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from gymnasium.wrappers import RecordVideo
from gymnasium.spaces import Box, Dict, Space
from gymnasium.envs.mujoco import MujocoEnv
from transformers import AutoTokenizer

# --- Configuration ---
# TODO: Replace with your specific Hugging Face repository IDs.
POLICY_REPO_ID = "jhou/smolvla_pickplace"
# TODO: Replace with the instruction for the task.
INSTRUCTION = "put the small object on the big object"
# Path to the MuJoCo XML file for the environment.
MODEL_XML_PATH = "mujoco-so101/so101-assets/so101_with_objects.xml"
#MODEL_XML_PATH = "mujoco-so101/so101-assets/so101_new_calib.xml"
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
        # Load the model once to get rendering dimensions from the XML
        model = mujoco.MjModel.from_xml_path(model_path)
        # Access the global visual settings using the 'global_' attribute
        width = model.vis.global_.offwidth
        height = model.vis.global_.offheight

        # Define the observation space to match SmolVLA's expected inputs.
        # It's a dictionary with 'images' and 'state'.
        observation_space = Dict({
            "images": Dict({
                "up": Box(
                    low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                ),
                "side": Box(
                    low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                ),
            }),
            "state": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
        })

        # We pass the model path, and gymnasium's MujocoEnv will load it again.
        # This is acceptable for this script's purpose.
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            width=width,
            height=height,
            **kwargs,
        )

    def _get_obs(self):
        """
        Returns the observation from the environment in the format expected by SmolVLA.
        """
        # Get the state vector (joint positions and velocities).
        state = np.concatenate([self.data.qpos[:6], self.data.qvel[:6]]).ravel()
        # Get the image observation from both cameras.
        image_up = self.mujoco_renderer.render(self.render_mode, camera_name="up")
        image_side = self.mujoco_renderer.render(self.render_mode, camera_name="side")
        return {"images": {"up": image_up, "side": image_side}, "state": state}

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
        self.data.ctrl[:] = action[:6]
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        observation = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

def get_cam_orientation(cam_pos, target_pos):
    """
    Computes the quaternion orientation for the camera to look towards the target position.
    
    :param cam_pos: numpy array of shape (3,) for camera position [x, y, z]
    :param target_pos: numpy array of shape (3,) for target position [x, y, z]
    :return: numpy array of shape (4,) for quaternion [w, x, y, z]
    """
    fwd_dir = target_pos - cam_pos
    fwd_dir = fwd_dir / np.linalg.norm(fwd_dir)

    Z = -fwd_dir
    world_up = np.array([0.0, 0.0, 1.0])

    X = np.cross(world_up, Z)
    X = X / np.linalg.norm(X)

    Y = np.cross(Z, X)

    mat = np.column_stack((X, Y, Z))

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())

    return quat


def main():
    """
    This script demonstrates how to use a pre-trained SmolVLA policy and its
    associated processors to run inference in a custom MuJoCo environment.
    """
    parser = argparse.ArgumentParser(description="Infer SO101 arm actions in response to a Mujoco env and record a video.")
    parser.add_argument(
        "--video-folder",
        type=Path,
        default="../media",
        help="Path to the folder to save the video.",
    )
    args = parser.parse_args()

    print(f"Loading policy from: {POLICY_REPO_ID}")
    device = "cpu"

    # 1. Initialize the custom MuJoCo environment to get its spaces.
    # Construct the absolute path to the model XML file.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_XML_PATH)
    print(f"Initializing custom MuJoCo environment from: {model_path}")
    env = SO101Env(model_path=str(model_path), render_mode="rgb_array", camera_name="side")

    side_camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
    up_camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "up")

    # 2. Create an EnvConfig from the environment instance.
    env_cfg = SO101EnvConfig(
        observation_space=env.observation_space,
        action_space=env.action_space,
        control_freq=env.metadata["render_fps"],
    )

    obs, info = env.reset()

    # set camera orientation
    # Orient towards position (0, 0, 0),
    target_pos = np.array([0, 0, 0]) # (0, 0, 0)
    env.model.cam_quat[side_camera_id] = get_cam_orientation(
            env.model.cam_pos[side_camera_id], target_pos)
    env.model.cam_quat[up_camera_id] = get_cam_orientation(
            env.model.cam_pos[up_camera_id], target_pos)

    # Hold a neutral position
    neutral_action = np.array([ 0.03755415, -1.7234037, 1.6718199, 1.2405578, -1.411793, 0.02459861])

    # Set the initial state of the robot to the neutral action pose while preserving object states
    neutral_qvel = np.zeros(6)
    # Get the full current state from the environment after reset
    current_qpos = env.data.qpos.copy()
    current_qvel = env.data.qvel.copy()
    # Overwrite the robot's part of the state with the neutral pose
    current_qpos[:6] = neutral_action
    current_qvel[:6] = neutral_qvel
    # Set the modified full state
    env.set_state(current_qpos, current_qvel)

    # Hold the neutral position for a moment to stabilize before running
    for _ in range(100):
        env.step(neutral_action)

    # Wrap the environment to record a video
    video_name_prefix = "rec"
    env = RecordVideo(env, str(args.video_folder), name_prefix=video_name_prefix)

    # 3. Load the policy configuration from the Hugging Face Hub.
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_REPO_ID)
    policy_cfg.pretrained_path = POLICY_REPO_ID
    policy_cfg.device = device

    # 4. Load the policy and processors.
    policy = make_policy(policy_cfg, env_cfg=env_cfg)
    pre_processor, post_processor = make_smolvla_pre_post_processors(config=policy_cfg)
    policy.eval()

    # Manually initialize the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    # 5. Run the inference loop.
    terminated = truncated = False
    step = 0

    print("Starting inference loop. Press Ctrl+C to exit.")
    try:
        while not terminated and not truncated:
            print("Step: {}".format(step))
            step = step + 1
            # a. Prepare the input for the pre-processor.
            # The pre-processor's _forward method expects a dictionary that looks like a Transition object.
            input_data = {
                "observation": obs,
                "action": None,
                "complementary_data": {"task": INSTRUCTION},
            }

            # b. Pre-process the observation, bypassing the batch converter.
            processed_batch = pre_processor._forward(input_data)

            # Manually tokenize the instruction and add it to the batch.
            tokenized_instruction = tokenizer(INSTRUCTION, return_tensors="pt")
            processed_batch["observation"]["language"] = {
                "tokens": tokenized_instruction["input_ids"],
                "attention_mask": tokenized_instruction["attention_mask"].bool(),
            }

            # Manually convert to tensor, add a batch dimension, and permute channels for the image.
            # The policy expects (b, c, h, w), but the output is (h, w, c).
            for cam_name in ["up", "side"]:
                img_array = processed_batch["observation"]["images"][cam_name]
                # Copy the array to resolve negative stride issues from the renderer.
                img_tensor = torch.from_numpy(img_array.copy())
                processed_batch["observation"]["images"][cam_name] = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            # Also convert the state to a batched tensor.
            state_array = processed_batch["observation"]["state"]
            processed_batch["observation"]["state"] = torch.from_numpy(state_array).unsqueeze(0).float()

            # c. Get the action from the policy.
            with torch.no_grad():
                flat_batch = flatten_dict(processed_batch)
                # The policy expects dot-separated keys, but flatten_dict uses slashes.
                flat_batch = {k.replace("/", "."): v for k, v in flat_batch.items()}
                action_prediction = policy.select_action(flat_batch)

            # d. Post-process the action.
            action_tensor = post_processor(action_prediction)
            action_env = action_tensor[0]
            print(action_env)

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
