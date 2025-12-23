#!/usr/bin/env python3
"""
Main script for replaying episodes from datasets or npy files in MuJoCo.

This script replays robot actions in a MuJoCo simulation environment, with support for:
- Loading data from HuggingFace datasets or local npy files
- Automatic grasp detection and object placement
- State comparison between simulation and real data
- Video recording of the replay
"""

import argparse
from pathlib import Path

from data_loader import validate_input_args, load_episode_data
from grasp_detection import detect_grasp_and_compute_object_pose
from environment import create_environment, place_object_in_scene
from replay_engine import generate_start_image, setup_video_recording, replay_actions_loop
from state_comparison import (
    validate_episode_data,
    setup_state_comparison,
    print_state_comparison_summary
)


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Replay an episode from a dataset or npy files in Mujoco and record a video."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID of the dataset to use.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="The episode index to process from the dataset.",
    )
    parser.add_argument(
        "--actions-npy-path",
        type=str,
        default=None,
        help="Path to npy file containing actions (alternative to dataset loading).",
    )
    parser.add_argument(
        "--states-npy-path",
        type=str,
        default=None,
        help="Path to npy file containing observation states (optional, needed for object placement).",
    )
    parser.add_argument(
        "--skip-object-placement",
        action="store_true",
        help="Skip automatic object placement (useful when states are not available).",
    )
    parser.add_argument(
        "--manual-object-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Manually specify object position [x y z] (skips grasp detection).",
    )
    parser.add_argument(
        "--robot-xml-file",
        type=str,
        default="so101-assets/so101_new_calib.xml",
        help="Path to the MuJoCo XML file for the robot model for FK.",
    )
    parser.add_argument(
        "--env-xml-file",
        type=str,
        default="so101-assets/so101_with_objects.xml",
        help="Path to the MuJoCo XML file for the environment with objects.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="object_to_grasp",
        help="Name of the object's body/joint in the XML file.",
    )
    parser.add_argument(
        "--video-folder",
        type=Path,
        default="../media",
        help="Path to the folder to save the video.",
    )
    parser.add_argument(
        "--start-image-only",
        action="store_true",
        help="Instead of writing a video, only write a snapshot image of the initial scene.",
    )
    parser.add_argument(
        "--compare-state",
        nargs='?',
        const='all',
        default=None,
        metavar='TIMESTEP',
        help="Compare the simulated actuator states with the dataset states. Use without argument to compare all timesteps, or provide a specific timestep number to compare only that timestep (with detailed vectors).",
    )
    parser.add_argument(
        "--fixed-steps",
        type=int,
        default=None,
        help="Execute each action for exactly N physics steps (instead of waiting for stabilization). Useful for matching real hardware timing.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        dest="video",
        default=True,
        help="Record video of the replay (default: yes).",
    )
    parser.add_argument(
        "--no-video",
        action="store_false",
        dest="video",
        help="Disable video recording.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Verbosity level for action execution progress (0=off, 1=on). Default: 1.",
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the replay script.
    """
    args = parse_arguments()

    # 1. Validate input arguments
    if not validate_input_args(args):
        return

    # 2. Load episode data (from npy files or HuggingFace dataset)
    episode = load_episode_data(args)
    if episode is None:
        return

    # 3. Detect grasp and compute object pose
    gripper_position, gripper_orientation_quat = detect_grasp_and_compute_object_pose(episode, args)
    if gripper_position is None and gripper_orientation_quat is None and not args.skip_object_placement:
        # Error occurred during grasp detection (not just skipped)
        if episode["observation.state"] is not None and not args.manual_object_position:
            return

    # 4. Create environment
    env, object_jnt_id, qpos_addr = create_environment(args.env_xml_file, "front_camera", args.object_name)
    if env is None:
        return

    # 5. Handle start image only mode
    if args.start_image_only:
        generate_start_image(env, episode, gripper_position, gripper_orientation_quat, qpos_addr, args)
        return

    # 6. Setup video recording
    env, video_name_prefix = setup_video_recording(env, args)

    # 7. Reset environment and place object
    env.reset()
    place_object_in_scene(env.unwrapped.data, qpos_addr, gripper_position, gripper_orientation_quat, args.object_name)

    # 8. Validate episode data
    actions, num_joints, valid = validate_episode_data(episode, env)
    if not valid:
        env.close()
        return

    # 9. Setup state comparison
    result = setup_state_comparison(args, episode)
    if result is None or result == (None, None, None, None, None):
        env.close()
        return
    compare_all_states, compare_specific_timestep, state_errors, fk_model, fk_data = result

    # 10. Replay actions
    replay_success = replay_actions_loop(
        env, episode, actions, num_joints, args,
        compare_all_states, compare_specific_timestep, state_errors,
        fk_model, fk_data
    )

    # 11. Cleanup
    if args.video:
        print("\nWriting video...")
    env.close()

    # 12. Print state comparison summary
    if compare_all_states:
        print_state_comparison_summary(state_errors, num_joints)

    # 13. Final message
    if args.video:
        print(f"\n\nSimulation finished. Video saved in the '{args.video_folder}' directory with prefix '{video_name_prefix}'.")
    else:
        print("\n\nSimulation finished.")


if __name__ == "__main__":
    main()
