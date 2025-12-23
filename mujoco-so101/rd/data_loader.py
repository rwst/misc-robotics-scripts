"""
Data loading utilities for episode data from various sources.
"""

import numpy as np
import datasets


def validate_input_args(args):
    """
    Validates command-line argument combinations.

    Args:
        args: Parsed command-line arguments

    Returns:
        bool: True if validation passes, False otherwise
    """
    if args.actions_npy_path and (args.repo_id or args.episode_index is not None):
        print("Error: Cannot specify both --actions-npy-path and dataset options (--repo-id, --episode-index)")
        return False
    if not args.actions_npy_path and not (args.repo_id and args.episode_index is not None):
        print("Error: Must specify either --actions-npy-path or both --repo-id and --episode-index")
        return False
    return True


def load_episode_data(args):
    """
    Loads episode data from either npy files or HuggingFace dataset.

    Args:
        args: Parsed command-line arguments with data source info

    Returns:
        dict: Episode dictionary with 'observation.state' and 'action' keys,
              or None if loading fails
    """
    episode = {"observation.state": None, "action": None}

    if args.actions_npy_path:
        # Load from npy files
        print(f"Loading actions from npy file: {args.actions_npy_path}")
        try:
            actions = np.load(args.actions_npy_path)
            episode["action"] = actions
            print(f"Actions loaded successfully. Shape: {actions.shape}")
        except Exception as e:
            print(f"Error loading actions from '{args.actions_npy_path}': {e}")
            return None

        if args.states_npy_path:
            print(f"Loading states from npy file: {args.states_npy_path}")
            try:
                states = np.load(args.states_npy_path)
                episode["observation.state"] = states
                print(f"States loaded successfully. Shape: {states.shape}")
            except Exception as e:
                print(f"Error loading states from '{args.states_npy_path}': {e}")
                return None
        else:
            print("No states file provided. Object placement will require manual position or will be skipped.")
    else:
        # Load from HuggingFace dataset
        try:
            print(f"Loading episode {args.episode_index} from dataset '{args.repo_id}'...")
            dataset = datasets.load_dataset(args.repo_id, split="train", streaming=True)
            episode_dataset = dataset.filter(
                lambda example: example["episode_index"] == args.episode_index
            )
            episode_steps = list(episode_dataset)

            if not episode_steps:
                print(f"Episode {args.episode_index} not found or is empty.")
                return None

            episode = {
                "observation.state": np.array(
                    [step["observation.state"] for step in episode_steps]
                ),
                "action": np.array([step["action"] for step in episode_steps]),
            }
            print("Episode loaded successfully.")
        except Exception as e:
            print(f"Error loading dataset '{args.repo_id}': {e}")
            return None

    return episode
