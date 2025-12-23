import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import mujoco
import datasets
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.wrappers import RecordVideo
from PIL import Image


class SO101Env(MujocoEnv):
    """
    Custom MuJoCo environment for the SO101 robot.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
    }

    def __init__(self, model_path, frame_skip=1, **kwargs):
        # The observation space consists of joint positions and velocities
        # This needs to be flexible based on the model.
        model = mujoco.MjModel.from_xml_path(model_path)
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(model.nq + model.nv,), dtype=np.float64
        )

        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            **kwargs,
        )

    def _get_obs(self):
        """
        Returns the observation from the environment.
        """
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

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
        self.data.ctrl[:action.shape[0]] = action
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        observation = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


def find_grasp_timestep(episode, gripper_joint_index):
    """
    Finds the timestep of the grasp event in an episode.
    """
    qpos = np.array(episode["observation.state"])
    gripper_qpos = qpos[:, gripper_joint_index]
    gripper_vel = np.diff(gripper_qpos, prepend=gripper_qpos[0])

    peak_closing_vel_idx = np.argmin(gripper_vel)

    for i in range(peak_closing_vel_idx + 1, len(gripper_vel)):
        if np.isclose(gripper_vel[i], 0, atol=1e-3):
            return i

    return None


def main():
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
    args = parser.parse_args()

    # Validate input arguments
    if args.actions_npy_path and (args.repo_id or args.episode_index is not None):
        print("Error: Cannot specify both --actions-npy-path and dataset options (--repo-id, --episode-index)")
        return
    if not args.actions_npy_path and not (args.repo_id and args.episode_index is not None):
        print("Error: Must specify either --actions-npy-path or both --repo-id and --episode-index")
        return

    # 1. Load the data (from npy files or dataset)
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
            return

        if args.states_npy_path:
            print(f"Loading states from npy file: {args.states_npy_path}")
            try:
                states = np.load(args.states_npy_path)
                episode["observation.state"] = states
                print(f"States loaded successfully. Shape: {states.shape}")
            except Exception as e:
                print(f"Error loading states from '{args.states_npy_path}': {e}")
                return
        else:
            print("No states file provided. Object placement will require manual position or will be skipped.")
    else:
        # Load from HuggingFace dataset (original behavior)
        try:
            print(f"Loading episode {args.episode_index} from dataset '{args.repo_id}'...")
            dataset = datasets.load_dataset(args.repo_id, split="train", streaming=True)
            episode_dataset = dataset.filter(
                lambda example: example["episode_index"] == args.episode_index
            )
            episode_steps = list(episode_dataset)

            if not episode_steps:
                print(f"Episode {args.episode_index} not found or is empty.")
                return

            episode = {
                "observation.state": np.array(
                    [step["observation.state"] for step in episode_steps]
                ),
                "action": np.array([step["action"] for step in episode_steps]),
            }
            print("Episode loaded successfully.")
        except Exception as e:
            print(f"Error loading dataset '{args.repo_id}': {e}")
            return

    # 2. Find the Grasp Pose using a temporary model (if needed)
    gripper_position = None
    gripper_orientation_quat = None

    if args.skip_object_placement:
        print("Skipping object placement (--skip-object-placement specified).")
    elif args.manual_object_position:
        print(f"Using manual object position: {args.manual_object_position}")
        gripper_position = np.array(args.manual_object_position)
        # Use default orientation (identity quaternion)
        gripper_orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])
    elif episode["observation.state"] is not None:
        # Automatic grasp detection using forward kinematics
        try:
            robot_model = mujoco.MjModel.from_xml_path(args.robot_xml_file)
            robot_data = mujoco.MjData(robot_model)
        except Exception as e:
            print(f"Error loading MuJoCo robot model from {args.robot_xml_file}: {e}")
            return

        gripper_joint_index = -1
        grasp_timestep = find_grasp_timestep(episode, gripper_joint_index)

        if grasp_timestep is None:
            print("Could not identify a clear grasp event in the episode.")
            print("Consider using --manual-object-position or --skip-object-placement")
            return

        print(f"Grasp event identified at timestep: {grasp_timestep}")

        grasp_qpos = episode["observation.state"][grasp_timestep]
        qpos_radians = np.deg2rad(grasp_qpos)
        robot_data.qpos[: len(qpos_radians)] = qpos_radians
        mujoco.mj_forward(robot_model, robot_data)

        try:
            gripper_site_name = "gripperframe"
            gripper_position = robot_data.site(gripper_site_name).xpos.copy()
            #gripper_position = gripper_position - 0.05
            gripper_orientation_mat = robot_data.site(gripper_site_name).xmat.reshape(3, 3).copy()
            gripper_orientation_quat = np.empty(4)
            mujoco.mju_mat2Quat(gripper_orientation_quat, gripper_orientation_mat.flatten())
            print(f"Estimated grasped object position: {gripper_position}")
        except KeyError:
            print(f"Error: Site '{gripper_site_name}' not found in the robot XML.")
            return
    else:
        print("WARNING: No states available for grasp detection and no manual position specified.")
        print("Object will not be placed. Use --manual-object-position or --skip-object-placement to suppress this warning.")

    # 3. Create the final environment
    env = SO101Env(
        model_path=os.path.abspath(args.env_xml_file),
        render_mode="rgb_array",
        camera_name="front_camera",
    )

    # 4. Get object joint info before wrapping (we'll place it after reset)
    object_jnt_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, args.object_name
    )
    qpos_addr = -1
    if object_jnt_id != -1:
        qpos_addr = env.model.jnt_qposadr[object_jnt_id]

    # Move the camera 30cm to the left
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
    if camera_id != -1:
        env.model.cam_pos[camera_id][0] -= 0.3

    if args.start_image_only:
        # Set initial robot state from episode FIRST
        if episode["observation.state"] is not None:
            initial_robot_qpos = np.deg2rad(episode["observation.state"][0])
            env.data.qpos[: len(initial_robot_qpos)] = initial_robot_qpos
            env.data.qvel[:] = 0  # Zero out all velocities

        # THEN place object (after setting robot state to avoid overwriting)
        if qpos_addr != -1 and gripper_position is not None:
            # Use gripper x,y but correct z-height for the object (0.025m for object1)
            object_position = gripper_position.copy()
            object_position[2] = 0.025  # Object z-height above ground
            env.data.qpos[qpos_addr : qpos_addr + 3] = object_position
            env.data.qpos[qpos_addr + 3 : qpos_addr + 7] = gripper_orientation_quat
            print(f"Placed object '{args.object_name}' at position: ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
            print(f"  (original gripper z was: {gripper_position[2]:.3f})")
            print(f"  qpos_addr: {qpos_addr}, total qpos size: {env.data.qpos.size}")
        elif qpos_addr != -1 and gripper_position is None:
            print(f"Object '{args.object_name}' found but no position available - object not placed")
        elif qpos_addr == -1:
            print(f"WARNING: Object '{args.object_name}' not found in XML!")

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
        return

    # 6. Wrap for video recording and replay actions
    if args.video:
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
    else:
        print("Video recording disabled.")

    env.reset()

    # Place object after reset (minimal approach - just set position)
    if qpos_addr != -1 and gripper_position is not None:
        # Use gripper x,y but correct z-height for the object (0.025m for object1)
        object_position = gripper_position.copy()
        object_position[2] = 0.025  # Object z-height above ground
        env.unwrapped.data.qpos[qpos_addr : qpos_addr + 3] = object_position
        env.unwrapped.data.qpos[qpos_addr + 3 : qpos_addr + 7] = gripper_orientation_quat
        print(f"Placed object '{args.object_name}' at grasp location.")
        print(f"  Position (x, y, z): ({object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f})")
        print(f"  (original gripper z was: {gripper_position[2]:.3f})")
    elif qpos_addr != -1 and gripper_position is None:
        print(f"Object '{args.object_name}' found but no position available - object not placed")

    actions = episode["action"]

    # Validate actions array
    if actions is None or len(actions) == 0:
        print("Error: No actions found in the episode data.")
        return

    if actions.ndim != 2:
        print(f"Error: Actions must be a 2D array, got shape {actions.shape}")
        return

    max_steps_per_move = 500
    movement_epsilon = 1e-3  # Stop if position change norm is less than this
    num_joints = env.action_space.shape[0]

    # Validate action dimension matches environment
    if actions.shape[1] != num_joints:
        print(f"Error: Action dimension ({actions.shape[1]}) does not match environment action space ({num_joints})")
        print(f"Actions shape: {actions.shape}")
        print(f"Environment action space: {env.action_space.shape}")
        return

    # Validate states shape if provided
    if episode["observation.state"] is not None:
        states = episode["observation.state"]
        if states.ndim != 2:
            print(f"Error: States must be a 2D array, got shape {states.shape}")
            return

        num_states = states.shape[0]
        num_actions = actions.shape[0]

        if num_states < num_actions:
            print(f"Error: Not enough states ({num_states}) for the number of actions ({num_actions})")
            print("Expected at least as many states as actions.")
            return
        elif num_states == num_actions:
            print(f"Warning: States and actions have the same length ({num_actions})")
            print("State comparison for the last action will not be available.")
        elif num_states > num_actions + 1:
            print(f"Warning: More states ({num_states}) than expected for {num_actions} actions")
            print(f"Expected {num_actions} or {num_actions + 1} states.")

    # For state comparison metrics
    compare_all_states = False
    compare_specific_timestep = None

    if args.compare_state is not None:
        if episode["observation.state"] is None:
            print("\n" + "="*80)
            print("WARNING: --compare-state specified but no state data available!")
            print("State comparison will be skipped. Provide --states-npy-path to enable comparison.")
            print("="*80)
        else:
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
                except ValueError:
                    print(f"Error: Invalid timestep value '{args.compare_state}'. Must be 'all' or an integer.")
                    return

    # Load FK model for gripper position computation if needed
    fk_model = None
    fk_data = None
    if compare_specific_timestep is not None:
        try:
            fk_model = mujoco.MjModel.from_xml_path(args.robot_xml_file)
            fk_data = mujoco.MjData(fk_model)
            print(f"Loaded FK model from {args.robot_xml_file} for gripper position computation")
        except Exception as e:
            print(f"Warning: Could not load FK model for gripper position computation: {e}")
            fk_model = None
            fk_data = None

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
                    try:
                        gripper_site_name = "gripperframe"

                        # Compute expected gripper position from dataset state
                        expected_qpos_rad = np.deg2rad(expected_state_deg)
                        fk_data.qpos[:len(expected_qpos_rad)] = expected_qpos_rad
                        mujoco.mj_forward(fk_model, fk_data)
                        expected_gripper_pos = fk_data.site(gripper_site_name).xpos.copy()

                        # Compute actual gripper position from simulated state
                        actual_qpos_rad = simulated_state_rad
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

                print("="*80)

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
            break

    if args.video:
        print("\nWriting video...")
    env.close()

    # Print state comparison summary (only for compare-all mode)
    if compare_all_states and state_errors:
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

    if args.video:
        print(f"\n\nSimulation finished. Video saved in the '{args.video_folder}' directory with prefix '{video_name_prefix}'.")
    else:
        print("\n\nSimulation finished.")


if __name__ == "__main__":
    main()
