# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains miscellaneous robotics scripts for working with the SO-ARM100 (SO101) robot, primarily focused on MuJoCo simulation, dataset processing, and policy inference using LeRobot models (ACT and SmolVLA). The robot is a 6-DoF arm with camera observations.

## Robot & Assets

- **Robot**: SO-ARM100 (SO101) - 6 degree-of-freedom robotic arm
- **Asset Sources**:
  - STL files: https://huggingface.co/haixuantao/dora-bambot/tree/main/URDF/assets
  - URDF/XML: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
- **Neutral Pose**: `[0.03755415, -1.7234037, 1.6718199, 1.2405578, -1.411793, 0.02459861]`
- **State Dimension**: 6 joint positions (state space is typically 12 when including velocities)

## MuJoCo Environment Architecture

### SO101Env Class

The core environment class (`SO101Env`) extends `gymnasium.envs.mujoco.MujocoEnv`. There are multiple variants throughout the codebase:

1. **Dynamic Size Version** (in `setup_env.py`, `sim_server.py`, `replay_dataset.py`): Observation space size determined dynamically from model (qpos.size + qvel.size)
2. **Fixed 12D Version** (in `replay_actions.py`): Hardcoded 12-dimensional observation space for 6-joint robot
3. **SmolVLA Version** (in `smolvla_mujoco_inference.py`): Returns Dict with `{"images": {"up": ..., "side": ...}, "state": ...}`

All versions return qpos + qvel concatenated via `_get_obs()`.

The environment includes utility functions for:
- AABB collision detection (`check_aabb_overlap`)
- Camera orientation computation (`get_cam_orientation`)
- Object randomization within sandbox (`randomize`)
- Setting camera orientations to look at target (`set_camera_orientations`)

### MuJoCo Models

Located in `mujoco-so101/so101-assets/`:
- `so101_new_calib.xml` - Basic calibrated SO101 model
- `so101_new_calib_black.xml` - Black background variant
- `so101_with_objects.xml` - Model with yellow cube ("object1") and blue tray ("object2")
- `so101_sim_server.xml` - For simulation server setup

Cameras:
- `front_camera`, `side_camera`, `top_camera`, `up` - Multiple view angles
- Camera dimensions read from XML global visual settings: `model.vis.global_.offwidth/offheight`

## Running Scripts

### Policy Inference

**ACT Policy (single-step)**:
```bash
python infer_single_step_act.py --policy_path funXedu/so101_act_lego_brick_v2
```

**SmolVLA Policy (single-step)**:
```bash
python infer_single_step_smolvla.py  # defaults to lerobot/smolvla_base
```

**SmolVLA with MuJoCo Loop**:
```bash
python smolvla_mujoco_inference.py --video-folder ../media
```
- Uses policy: `jhou/smolvla_pickplace`
- Default instruction: "put the small object on the big object"
- Records video to media folder

### MuJoCo Simulation

**Environment Setup/Snapshot**:
```bash
# Basic snapshot
python mujoco-so101/setup_env.py --model-path mujoco-so101/so101-assets/so101_with_objects.xml --image-name output.png

# With randomized objects
python mujoco-so101/setup_env.py --model-path mujoco-so101/so101-assets/so101_with_objects.xml --random --image-name random_scene.png
```

**Replay Actions**:
```bash
# Generate video (uses so101_new_calib.xml and front_camera, moved 30cm left)
python mujoco-so101/replay_actions.py --actions-path episode_0_actions.npy --video-folder ../media

# Just snapshot
python mujoco-so101/replay_actions.py --actions-path episode_0_actions.npy --start-image-only

# Actions are scaled: action / 100 * (action_space.high or -action_space.low based on sign)
```

**Test Actuators**:
```bash
python mujoco-so101/test_actuators.py  # Tests each joint sequentially
```

### Dataset Processing

**Export State Data**:
```bash
python mujoco-so101/export_state_data.py --repo-id <dataset_repo> --episode-index 0 --output-path episode_0_states.npy
# Optional: --root <local_path> --batch-size 32 --num-workers 4 --tolerance-s 1e-4
```

**Export Action Data**:
```bash
python mujoco-so101/export_action_data.py --repo-id <dataset_repo> --episode-index 0 --output-path episode_0_actions.npy
# Optional: --root <local_path> --batch-size 32 --num-workers 4 --tolerance-s 1e-4
```

**Replay Dataset Episode with Object Placement**:
```bash
python mujoco-so101/replay_dataset.py --repo-id <dataset_repo> --episode-index 0 --video-folder ../media
# Uses FK to infer grasped object position and places it in the scene
# Optional: --start-image-only, --robot-xml-file, --env-xml-file, --object-name
```

**Download HuggingFace Dataset**:
```bash
python hf_get_dataset.py  # Wrapper around snapshot_download()
```

### Utilities

**Inspect NumPy Files**:
```bash
python inspect_npy.py <path_to_file.npy>  # Wrapper around numpy.load()
```

## Key Implementation Details

### Policy Loading

Both ACT and SmolVLA policies require:
1. Loading `PreTrainedConfig` from HuggingFace Hub
2. Creating a dummy/real `EnvConfig` with observation and action spaces
3. Using `make_policy()` from `lerobot.policies.factory`
4. The ACT single-step script requires PR #1771 to be merged into lerobot

### SmolVLA Pre/Post Processing

- Use `make_smolvla_pre_post_processors()` from `lerobot.policies.smolvla.processor_smolvla`
- Manually tokenize instructions with `AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")`
- Images must be: numpy array → tensor → add batch dim → permute to (b, c, h, w)
- Flatten observation dict with `flatten_dict()` and replace "/" with "."

### Action Execution Pattern

When replaying actions in simulation, use stabilization loop:
```python
max_steps_per_move = 100-500
movement_epsilon = 1e-3 to 1e-4
previous_pos = np.full(num_joints, np.inf)
for substep in range(max_steps_per_move):
    obs, reward, terminated, truncated, info = env.step(action)
    current_pos = obs[:num_joints]  # or obs['state'][:num_joints]
    norm = np.linalg.norm(current_pos - previous_pos)
    if norm < movement_epsilon:
        break
    previous_pos = current_pos
```

### Randomization Algorithm

The `randomize()` function (in `setup_env.py`, `sim_server.py`) implements a collision-free object placement algorithm:
1. Place object2 (blue tray/larger object) randomly in sandbox with random orientation
2. Check if overlaps with arm base (5cm x 5cm at origin) → retry if yes
3. Place object1 (yellow cube/smaller object) randomly with random orientation
4. Check if overlaps with object2 or arm base → restart from step 1 if yes
5. Sandbox bounds: origin at arm base (0,0), width=0.4m, height=0.8m
6. Object dimensions: object1=2cm×1cm at z=0.025m, object2=10cm×15cm at z=0.01m
7. Updates qpos with new positions/quaternions and sets state

For `sim_server.py` variant: also randomizes side camera position on semicircle (x>0, distance 0.8m from origin, z=0.4m)

## Key Scripts and Files

Scripts are located in two places:
- **Root directory**: Inference scripts (`infer_single_step_*.py`, `smolvla_mujoco_inference.py`) and utilities (`hf_get_dataset.py`, `inspect_npy.py`)
- **mujoco-so101/**: Simulation scripts (`setup_env.py`, `replay_*.py`, `export_*.py`, `test_actuators.py`, `sim_server.py`)
- **mujoco-so101/so101-assets/**: MuJoCo XML models and STL files

## Dependencies

Main libraries:
- `mujoco` - Physics simulation
- `gymnasium` - RL environment interface
- `lerobot` - Robot learning models and datasets (LeRobotDataset, policies, etc.)
- `torch` - Deep learning
- `transformers` - For SmolVLA tokenizer
- `draccus` - Configuration management (for ACT script)
- `PIL` - Image processing
- `numpy` - Numerical operations
- `datasets` - HuggingFace datasets library (for `replay_dataset.py`)
