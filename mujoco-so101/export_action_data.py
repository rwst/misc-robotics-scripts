#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export action data for a given episode from a LeRobotDataset to a numpy file."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.visualize_dataset import EpisodeSampler


def export_episode_actions(
    dataset: LeRobotDataset,
    episode_index: int,
    output_path: Path,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Export actions from a single episode to a .npy file."""
    logging.info(f"Exporting actions for episode {episode_index} to {output_path}")

    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    actions = []
    indices = []
    for batch in tqdm.tqdm(dataloader, desc="Extracting actions"):
        if "action" in batch:
            actions.append(batch["action"])
            indices.append(batch["index"])

    if not actions:
        logging.warning("No actions found in the episode.")
        return

    # Sort actions by index to ensure correct order, which is important
    # when using multiple workers for data loading.
    indices = torch.cat(indices)
    actions = torch.cat(actions)

    sorted_indices = torch.argsort(indices)
    sorted_actions = actions[sorted_indices]

    actions_np = sorted_actions.numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, actions_np)
    logging.info(f"Successfully saved {len(actions_np)} actions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export action data from a LeRobotDataset episode to a .npy file for Mujoco replay."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to export actions from.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the output .npy file.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes for Dataloader.",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value. "
            "This is an argument passed to the constructor of LeRobotDataset."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info("Loading dataset")
    dataset = LeRobotDataset(args.repo_id, root=args.root, tolerance_s=args.tolerance_s)

    export_episode_actions(
        dataset=dataset,
        episode_index=args.episode_index,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
