"""
Action replay engine and execution logic.
"""

import mujoco
import numpy as np
from pathlib import Path
from PIL import Image
from gymnasium.wrappers import RecordVideo

from state_comparison import print_detailed_state_comparison


def generate_start_image(env, episode, gripper_position, gripper_orientation_quat, qpos_addr, args):
    """
    Generates a snapshot image of the initial scene and saves it.

    Args:
        env: MuJoCo environment
        episode: Episode dictionary with state data
        gripper_position: Gripper position for object placement
        gripper_orientation_quat: Gripper orientation quaternion
        qpos_addr: qpos address for object joint
        args: Command-line arguments
    """
    from environment import place_object_in_scene

    # Set initial robot state from episode FIRST
    if episode["observation.state"] is not None:
        initial_robot_qpos = np.deg2rad(episode["observation.state"][0])
        env.data.qpos[: len(initial_robot_qpos)] = initial_robot_qpos
        env.data.qvel[:] = 0  # Zero out all velocities

    # THEN place object (after setting robot state to avoid overwriting)
    place_object_in_scene(env.data, qpos_addr, gripper_position, gripper_orientation_quat, args.object_name, env.model)

    # Update physics to reflect the new state
    mujoco.mj_forward(env.model, env.data)

    print("Generating start image only...")
    frame = env.render()
    image = Image.fromarray(frame)

    # Generate filename based on input source
    if args.actions_npy_path:
        actions_basename = Path(args.actions_npy_path).stem
        output_path = args.video_folder / f"replay_{actions_basename}_start.png"
    else:
        output_path = (
            args.video_folder
            / f"replay_{args.repo_id.replace('/', '_')}_ep{args.episode_index}_start.png"
        )
    args.video_folder.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Start image saved to {output_path}")
    env.close()


def setup_video_recording(env, args):
    """
    Wraps environment with RecordVideo if video recording is enabled.

    Args:
        env: MuJoCo environment
        args: Command-line arguments

    Returns:
        tuple: (wrapped_env, video_name_prefix)
    """
    if not args.video:
        print("Video recording disabled.")
        return env, None

    # Generate video name based on input source
    if args.actions_npy_path:
        actions_basename = Path(args.actions_npy_path).stem
        video_name_prefix = f"replay_{actions_basename}"
    else:
        video_name_prefix = (
            f"replay_{args.repo_id.replace('/', '_')}_ep{args.episode_index}"
        )

    env = RecordVideo(env, str(args.video_folder), name_prefix=video_name_prefix)
    print(f"Video recording enabled. Output will be saved with prefix '{video_name_prefix}'")

    return env, video_name_prefix


def replay_actions_loop(env, episode, actions, num_joints, args, compare_all_states,
                       compare_specific_timestep, state_errors, fk_model, fk_data):
    """
    Main action replay loop with optional state comparison.

    Args:
        env: MuJoCo environment
        episode: Episode dictionary with state data
        actions: Array of actions to replay
        num_joints: Number of joints in the robot
        args: Command-line arguments
        compare_all_states: Whether to compare all timesteps
        compare_specific_timestep: Specific timestep to compare (or None)
        state_errors: List to collect state errors
        fk_model: Forward kinematics model (or None)
        fk_data: Forward kinematics data (or None)

    Returns:
        bool: True if replay completed successfully, False if terminated early
    """
    max_steps_per_move = 500
    movement_epsilon = 1e-3

    # Print execution mode
    if args.fixed_steps:
        print("\n" + "="*80)
        print(f"FIXED-STEP MODE: Executing {args.fixed_steps} physics steps per action")
        print("="*80)

    for i, action in enumerate(actions):
        # IMPORTANT: Actions from LeRobot datasets are already in degrees for SO101
        # Convert directly to radians without the -100 to 100 scaling
        scaled_action = np.deg2rad(action)

        if args.fixed_steps:
            # Fixed-duration execution (matches real hardware timing)
            for step in range(args.fixed_steps):
                if args.verbosity > 0:
                    print(f"\rExecuting action {i+1}/{len(actions)}... (step {step + 1}/{args.fixed_steps})", end='', flush=True)
                observation, reward, terminated, truncated, info = env.step(scaled_action)

                if terminated or truncated:
                    if args.verbosity > 0:
                        print(f"\rExecuting action {i+1}/{len(actions)}... terminated/truncated at step {step + 1}.                    ", end='', flush=True)
                    break
            else:
                if args.verbosity > 0:
                    print(f"\rExecuting action {i+1}/{len(actions)}... completed {args.fixed_steps} steps.                    ", end='', flush=True)
        else:
            # Stabilization mode (wait until movement stops)
            previous_pos = np.full(num_joints, np.inf)
            for step in range(max_steps_per_move):
                # Print progress on same line
                if args.verbosity > 0:
                    print(f"\rExecuting action {i+1}/{len(actions)}... (substep {step + 1}/{max_steps_per_move})", end='', flush=True)

                observation, reward, terminated, truncated, info = env.step(scaled_action)
                current_pos = observation[:num_joints]

                # Check if the movement has stopped
                norm = np.linalg.norm(current_pos - previous_pos)
                if norm < movement_epsilon:
                    if args.verbosity > 0:
                        print(f"\rExecuting action {i+1}/{len(actions)}... stabilized in {step + 1} steps.                    ", end='', flush=True)
                    break

                previous_pos = current_pos

                if terminated or truncated:
                    if args.verbosity > 0:
                        print(f"\rExecuting action {i+1}/{len(actions)}... terminated/truncated at step {step + 1}.                    ", end='', flush=True)
                    break
            else:
                # This block executes if the for loop completes without a 'break'
                if args.verbosity > 0:
                    print(f"\rExecuting action {i+1}/{len(actions)}... WARNING: timed out after {max_steps_per_move} steps (norm={norm:.6f}).                    ", end='', flush=True)

        # Compare simulated state with dataset state
        if episode["observation.state"] is not None and i + 1 < len(episode["observation.state"]):
            # Get current simulated state (convert from radians to degrees)
            simulated_state_rad = observation[:num_joints]
            simulated_state_deg = np.rad2deg(simulated_state_rad)

            # Get expected state from dataset (next timestep)
            expected_state_deg = episode["observation.state"][i + 1]

            # Detailed comparison for specific timestep
            if compare_specific_timestep is not None and i == compare_specific_timestep:
                print_detailed_state_comparison(
                    i, action, expected_state_deg, simulated_state_deg,
                    num_joints, fk_model, fk_data
                )

            # Collect statistics for all timesteps mode
            if compare_all_states:
                # Calculate errors
                absolute_errors = np.abs(simulated_state_deg - expected_state_deg)
                mae = np.mean(absolute_errors)
                rmse = np.sqrt(np.mean((simulated_state_deg - expected_state_deg)**2))
                max_error = np.max(absolute_errors)

                state_errors.append({
                    'timestep': i,
                    'mae': mae,
                    'rmse': rmse,
                    'max_error': max_error,
                    'per_joint_errors': absolute_errors
                })

                if args.verbosity > 0:
                    print(f"\n  State comparison [timestep {i}→{i+1}]: MAE={mae:.4f}°, RMSE={rmse:.4f}°, Max={max_error:.4f}° (joint {np.argmax(absolute_errors)})")

        if terminated or truncated:
            print("\n\nReplay finished due to episode termination.")
            return False

    return True
