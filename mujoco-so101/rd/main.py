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
import mujoco
import numpy as np
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
        default="so101-assets/so101_new_calib_black.xml",
        help="Path to the MuJoCo XML file for the environment. Use so101_new_calib_black.xml (default, no objects) for accurate replay, or so101_with_objects.xml for visualization with objects.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="object1_to_world",
        help="Name of the object's joint in the XML file (e.g., 'object1_to_world' or 'object2_to_world').",
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
    parser.add_argument(
        "--video-threads",
        type=int,
        default=4,
        help="Number of threads for video encoding (default: 4).",
    )
    parser.add_argument(
        "--keep-nth-video-frame",
        type=int,
        default=1,
        help="Keep every Nth frame for video (1=all frames, 10=every 10th). Default: 1 (all frames).",
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

    # DEBUG: Log what was returned from grasp detection
    if gripper_position is not None:
        print(f"[MAIN DEBUG] Returned from detect_grasp_and_compute_object_pose:")
        print(f"[MAIN DEBUG]   gripper_position = {gripper_position}")
        print(f"[MAIN DEBUG]   gripper_orientation_quat = {gripper_orientation_quat}")

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

    # 6. Setup video recording (returns encoder instead of wrapping env)
    env, encoder, video_name = setup_video_recording(env, args)

    # 7. Reset environment and place object
    env.reset()

    # CRITICAL: Set initial robot state from episode data BEFORE replaying actions
    # Without this, the robot starts from neutral pose [0,0,0,0,0,0] instead of
    # the actual starting pose from the dataset, causing large errors
    # IMPORTANT: Use env.unwrapped to bypass any wrappers and access the actual MuJoCo env
    if episode["observation.state"] is not None:
        initial_robot_qpos = np.deg2rad(episode["observation.state"][0])
        env.unwrapped.data.qpos[: len(initial_robot_qpos)] = initial_robot_qpos
        env.unwrapped.data.qvel[:] = 0  # Zero out all velocities

    # Place object AFTER setting robot state (to avoid overwriting robot qpos)
    print(f"[MAIN DEBUG] About to call place_object_in_scene with:")
    print(f"[MAIN DEBUG]   gripper_position = {gripper_position}")
    print(f"[MAIN DEBUG]   gripper_orientation_quat = {gripper_orientation_quat}")
    place_object_in_scene(env.unwrapped.data, qpos_addr, gripper_position, gripper_orientation_quat, args.object_name, env.unwrapped.model)

    # Update physics to reflect the new state
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

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

    # DEBUG: Print current state before replay
    print(f"\nDEBUG: State just before replay_actions_loop:")
    print(f"  env.unwrapped.data.qpos[:6] (deg) = {np.rad2deg(env.unwrapped.data.qpos[:6])}")
    print(f"  Expected initial state  = {episode['observation.state'][0]}")
    print(f"  Match: {np.allclose(np.rad2deg(env.unwrapped.data.qpos[:6]), episode['observation.state'][0], atol=0.01)}")
    print()

    # 10. Replay actions (pass encoder for video recording)
    replay_success = replay_actions_loop(
        env, episode, actions, num_joints, args,
        compare_all_states, compare_specific_timestep, state_errors,
        fk_model, fk_data, encoder=encoder
    )

    # 11. Cleanup
    env.close()
    if encoder:
        print("\nFinalizing video encoding...")
        frame_count = encoder.close()
        print(f"Video encoded: {frame_count} frames")

    # 12. Print state comparison summary
    if compare_all_states:
        print_state_comparison_summary(state_errors, num_joints)

    # 13. Final message
    if encoder:
        print(f"\n\nSimulation finished. Video saved: {args.video_folder / video_name}")
    else:
        print("\n\nSimulation finished.")


if __name__ == "__main__":
    main()
