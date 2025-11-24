# replay_dataset.py - NPY File Support

The `replay_dataset.py` script now supports loading actions (and optionally states) from local `.npy` files in addition to HuggingFace datasets.

## Usage Modes

### Mode 1: Load from HuggingFace Dataset (Original)
```bash
python replay_dataset.py \
    --repo-id <dataset_repo> \
    --episode-index 0 \
    --video-folder ../media
```

### Mode 2: Load from NPY files with Object Placement
When you have both actions and states saved as npy files:
```bash
python replay_dataset.py \
    --actions-npy-path episode_0_actions.npy \
    --states-npy-path episode_0_states.npy \
    --video-folder ../media
```
This mode will:
- Load actions from the npy file
- Load states from the npy file
- Use forward kinematics to detect grasp position
- Place object at the detected position

### Mode 3: Actions Only (No Object Placement)
When you only have actions:
```bash
python replay_dataset.py \
    --actions-npy-path episode_0_actions.npy \
    --skip-object-placement \
    --video-folder ../media
```

### Mode 4: Actions with Manual Object Position
Specify object position manually:
```bash
python replay_dataset.py \
    --actions-npy-path episode_0_actions.npy \
    --manual-object-position 0.2 0.1 0.025 \
    --video-folder ../media
```

## New Arguments

- `--actions-npy-path PATH`: Path to npy file containing actions (shape: `[timesteps, action_dim]`)
- `--states-npy-path PATH`: Path to npy file containing observation states (shape: `[timesteps, state_dim]`)
- `--skip-object-placement`: Skip automatic object placement entirely
- `--manual-object-position X Y Z`: Manually specify object position in meters

## Additional Options

### Generate Start Image Only
```bash
python replay_dataset.py \
    --actions-npy-path episode_0_actions.npy \
    --skip-object-placement \
    --start-image-only \
    --video-folder ../media
```

### Disable Video Recording
```bash
python replay_dataset.py \
    --actions-npy-path episode_0_actions.npy \
    --skip-object-placement \
    --no-video
```

### State Comparison (requires states)
```bash
python replay_dataset.py \
    --actions-npy-path episode_0_actions.npy \
    --states-npy-path episode_0_states.npy \
    --compare-state \
    --video-folder ../media
```

### Fixed-Step Execution
Execute each action for exactly N physics steps instead of waiting for stabilization:
```bash
python replay_dataset.py \
    --actions-npy-path episode_0_actions.npy \
    --skip-object-placement \
    --fixed-steps 50 \
    --video-folder ../media
```

## Exporting NPY Files

You can export actions and states from HuggingFace datasets using:

```bash
# Export actions
python export_action_data.py \
    --repo-id <dataset_repo> \
    --episode-index 0 \
    --output-path episode_0_actions.npy

# Export states
python export_state_data.py \
    --repo-id <dataset_repo> \
    --episode-index 0 \
    --output-path episode_0_states.npy
```

## Notes

- When using `--actions-npy-path`, you cannot specify `--repo-id` or `--episode-index` (they are mutually exclusive)
- Actions should be in the same format as dataset actions (typically shape `[timesteps, 6]` for SO101)
- States should be in degrees (same format as dataset states)
- Object placement via grasp detection requires states data
- Video files will be named based on the npy filename (e.g., `replay_episode_0_actions-XXXXX.mp4`)
