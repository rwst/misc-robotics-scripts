"""
State validation and comparison utilities.
"""

import mujoco
import numpy as np


def validate_episode_data(episode, env):
    """
    Validates episode data shapes and compatibility with environment.

    Args:
        episode: Episode dictionary with action and state data
        env: MuJoCo environment

    Returns:
        tuple: (actions, num_joints, valid) where valid is True if validation passes
    """
    actions = episode["action"]

    # Validate actions array
    if actions is None or len(actions) == 0:
        print("Error: No actions found in the episode data.")
        return None, None, False

    if actions.ndim != 2:
        print(f"Error: Actions must be a 2D array, got shape {actions.shape}")
        return None, None, False

    num_joints = env.action_space.shape[0]

    # Validate action dimension matches environment
    if actions.shape[1] != num_joints:
        print(f"Error: Action dimension ({actions.shape[1]}) does not match environment action space ({num_joints})")
        print(f"Actions shape: {actions.shape}")
        print(f"Environment action space: {env.action_space.shape}")
        return None, None, False

    # Validate states shape if provided
    if episode["observation.state"] is not None:
        states = episode["observation.state"]
        if states.ndim != 2:
            print(f"Error: States must be a 2D array, got shape {states.shape}")
            return None, None, False

        num_states = states.shape[0]
        num_actions = actions.shape[0]

        if num_states < num_actions:
            print(f"Error: Not enough states ({num_states}) for the number of actions ({num_actions})")
            print("Expected at least as many states as actions.")
            return None, None, False
        elif num_states == num_actions:
            print(f"Warning: States and actions have the same length ({num_actions})")
            print("State comparison for the last action will not be available.")
        elif num_states > num_actions + 1:
            print(f"Warning: More states ({num_states}) than expected for {num_actions} actions")
            print(f"Expected {num_actions} or {num_actions + 1} states.")

    return actions, num_joints, True


def setup_state_comparison(args, episode):
    """
    Sets up state comparison mode based on command-line arguments.

    Args:
        args: Command-line arguments
        episode: Episode dictionary with state data

    Returns:
        tuple: (compare_all_states, compare_specific_timestep, state_errors, fk_model, fk_data)
    """
    compare_all_states = False
    compare_specific_timestep = None
    state_errors = []
    fk_model = None
    fk_data = None

    if args.compare_state is None:
        return compare_all_states, compare_specific_timestep, state_errors, fk_model, fk_data

    if episode["observation.state"] is None:
        print("\n" + "="*80)
        print("WARNING: --compare-state specified but no state data available!")
        print("State comparison will be skipped. Provide --states-npy-path to enable comparison.")
        print("="*80)
        return compare_all_states, compare_specific_timestep, state_errors, fk_model, fk_data

    if args.compare_state == 'all':
        compare_all_states = True
        state_errors = []
        print("\n" + "="*80)
        print("STATE COMPARISON ENABLED (all timesteps)")
        print("="*80)
    else:
        try:
            compare_specific_timestep = int(args.compare_state)
            print("\n" + "="*80)
            print(f"STATE COMPARISON ENABLED (timestep {compare_specific_timestep} only)")
            print("="*80)

            # Load FK model for gripper position computation
            try:
                fk_model = mujoco.MjModel.from_xml_path(args.robot_xml_file)
                fk_data = mujoco.MjData(fk_model)
                print(f"Loaded FK model from {args.robot_xml_file} for gripper position computation")
            except Exception as e:
                print(f"Warning: Could not load FK model for gripper position computation: {e}")
                fk_model = None
                fk_data = None
        except ValueError:
            print(f"Error: Invalid timestep value '{args.compare_state}'. Must be 'all' or an integer.")
            return None, None, None, None, None

    return compare_all_states, compare_specific_timestep, state_errors, fk_model, fk_data


def print_detailed_state_comparison(i, action, expected_state_deg, simulated_state_deg,
                                     num_joints, fk_model, fk_data):
    """
    Prints detailed state comparison for a specific timestep.

    Args:
        i: Current timestep index
        action: Action taken at timestep i
        expected_state_deg: Expected joint states in degrees
        simulated_state_deg: Simulated joint states in degrees
        num_joints: Number of joints
        fk_model: Forward kinematics model (or None)
        fk_data: Forward kinematics data (or None)
    """
    print("\n" + "="*80)
    print(f"DETAILED STATE COMPARISON AT TIMESTEP {i}")
    print("="*80)

    # Print action vector (already in degrees from dataset)
    print(f"\nAction vector [timestep {i}] (degrees):")
    print(f"  {action}")

    # Print expected state from dataset
    print(f"\nExpected state [timestep {i+1}] from dataset (degrees):")
    print(f"  {expected_state_deg}")

    # Print actual simulated state
    print(f"\nActual MuJoCo state [timestep {i+1}] (degrees):")
    print(f"  {simulated_state_deg}")

    # Print per-joint differences
    absolute_errors = np.abs(simulated_state_deg - expected_state_deg)
    print(f"\nPer-joint errors (degrees):")
    for joint_idx in range(num_joints):
        print(f"  Joint {joint_idx}: {absolute_errors[joint_idx]:.4f}°")

    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean((simulated_state_deg - expected_state_deg)**2))
    max_error = np.max(absolute_errors)
    print(f"\nAggregate errors:")
    print(f"  MAE:  {mae:.4f}°")
    print(f"  RMSE: {rmse:.4f}°")
    print(f"  Max:  {max_error:.4f}° (joint {np.argmax(absolute_errors)})")

    # Compute gripper positions if FK model is available
    if fk_model is not None and fk_data is not None:
        print_gripper_position_comparison(
            expected_state_deg, simulated_state_deg, fk_model, fk_data
        )

    print("="*80)


def print_gripper_position_comparison(expected_state_deg, simulated_state_deg, fk_model, fk_data):
    """
    Computes and prints gripper position comparison using FK.

    Args:
        expected_state_deg: Expected joint states in degrees
        simulated_state_deg: Simulated joint states in degrees
        fk_model: Forward kinematics MuJoCo model
        fk_data: Forward kinematics MuJoCo data
    """
    try:
        gripper_site_name = "gripperframe"

        # Compute expected gripper position from dataset state
        expected_qpos_rad = np.deg2rad(expected_state_deg)
        fk_data.qpos[:len(expected_qpos_rad)] = expected_qpos_rad
        mujoco.mj_forward(fk_model, fk_data)
        expected_gripper_pos = fk_data.site(gripper_site_name).xpos.copy()

        # Compute actual gripper position from simulated state
        actual_qpos_rad = np.deg2rad(simulated_state_deg)
        fk_data.qpos[:len(actual_qpos_rad)] = actual_qpos_rad
        mujoco.mj_forward(fk_model, fk_data)
        actual_gripper_pos = fk_data.site(gripper_site_name).xpos.copy()

        # Compute position difference
        gripper_pos_error = np.linalg.norm(actual_gripper_pos - expected_gripper_pos)
        gripper_pos_diff = actual_gripper_pos - expected_gripper_pos

        print(f"\nGripper position comparison:")
        print(f"  Expected (from dataset):    [{expected_gripper_pos[0]:.6f}, {expected_gripper_pos[1]:.6f}, {expected_gripper_pos[2]:.6f}] m")
        print(f"  Actual (from simulation):   [{actual_gripper_pos[0]:.6f}, {actual_gripper_pos[1]:.6f}, {actual_gripper_pos[2]:.6f}] m")
        print(f"  Euclidean error: {gripper_pos_error:.6f} m ({gripper_pos_error*1000:.3f} mm)")
        print(f"  Per-axis differences: dx={gripper_pos_diff[0]:.6f} m, dy={gripper_pos_diff[1]:.6f} m, dz={gripper_pos_diff[2]:.6f} m")
    except KeyError:
        print(f"\nWarning: Site '{gripper_site_name}' not found in FK model - cannot compute gripper positions")
    except Exception as e:
        print(f"\nWarning: Could not compute gripper positions: {e}")


def print_state_comparison_summary(state_errors, num_joints):
    """
    Prints aggregate statistics for state comparison across all timesteps.

    Args:
        state_errors: List of error dictionaries from all timesteps
        num_joints: Number of joints in the robot
    """
    if not state_errors:
        return

    print("\n" + "="*80)
    print("STATE COMPARISON SUMMARY")
    print("="*80)

    all_mae = [e['mae'] for e in state_errors]
    all_rmse = [e['rmse'] for e in state_errors]
    all_max = [e['max_error'] for e in state_errors]

    print(f"Overall statistics across {len(state_errors)} timesteps:")
    print(f"  Mean Absolute Error (MAE):")
    print(f"    Mean:   {np.mean(all_mae):.4f}°")
    print(f"    Median: {np.median(all_mae):.4f}°")
    print(f"    Std:    {np.std(all_mae):.4f}°")
    print(f"    Min:    {np.min(all_mae):.4f}°")
    print(f"    Max:    {np.max(all_mae):.4f}°")
    print(f"\n  Root Mean Square Error (RMSE):")
    print(f"    Mean:   {np.mean(all_rmse):.4f}°")
    print(f"    Median: {np.median(all_rmse):.4f}°")
    print(f"    Std:    {np.std(all_rmse):.4f}°")
    print(f"    Min:    {np.min(all_rmse):.4f}°")
    print(f"    Max:    {np.max(all_rmse):.4f}°")
    print(f"\n  Maximum per-joint error:")
    print(f"    Mean:   {np.mean(all_max):.4f}°")
    print(f"    Median: {np.median(all_max):.4f}°")
    print(f"    Max:    {np.max(all_max):.4f}°")

    # Per-joint statistics
    print(f"\n  Per-joint error statistics (MAE across all timesteps):")
    all_per_joint = np.array([e['per_joint_errors'] for e in state_errors])
    for joint_idx in range(num_joints):
        joint_errors = all_per_joint[:, joint_idx]
        print(f"    Joint {joint_idx}: Mean={np.mean(joint_errors):.4f}°, Std={np.std(joint_errors):.4f}°, Max={np.max(joint_errors):.4f}°")

    print("="*80)
