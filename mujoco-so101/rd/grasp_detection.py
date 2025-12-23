"""
Grasp detection and forward kinematics utilities.
"""

import mujoco
import numpy as np


def find_grasp_timestep(episode, gripper_joint_index):
    """
    Finds the timestep of the grasp event in an episode.

    Args:
        episode: Episode dictionary with 'observation.state' key
        gripper_joint_index: Index of the gripper joint in qpos

    Returns:
        int: Timestep of grasp event, or None if not found
    """
    qpos = np.array(episode["observation.state"])
    gripper_qpos = qpos[:, gripper_joint_index]
    gripper_vel = np.diff(gripper_qpos, prepend=gripper_qpos[0])

    peak_closing_vel_idx = np.argmin(gripper_vel)

    for i in range(peak_closing_vel_idx + 1, len(gripper_vel)):
        if np.isclose(gripper_vel[i], 0, atol=1e-3):
            return i

    return None


def detect_grasp_and_compute_object_pose(episode, args):
    """
    Detects grasp event and computes object position/orientation using FK.

    Args:
        episode: Episode dictionary with state data
        args: Command-line arguments with grasp detection settings

    Returns:
        tuple: (gripper_position, gripper_orientation_quat) or (None, None) if skipped/failed
    """
    gripper_position = None
    gripper_orientation_quat = None

    if args.skip_object_placement:
        print("Skipping object placement (--skip-object-placement specified).")
        return None, None

    if args.manual_object_position:
        print(f"Using manual object position: {args.manual_object_position}")
        gripper_position = np.array(args.manual_object_position)
        gripper_orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])
        return gripper_position, gripper_orientation_quat

    if episode["observation.state"] is None:
        print("WARNING: No states available for grasp detection and no manual position specified.")
        print("Object will not be placed. Use --manual-object-position or --skip-object-placement to suppress this warning.")
        return None, None

    # Automatic grasp detection using forward kinematics
    try:
        robot_model = mujoco.MjModel.from_xml_path(args.robot_xml_file)
        robot_data = mujoco.MjData(robot_model)
    except Exception as e:
        print(f"Error loading MuJoCo robot model from {args.robot_xml_file}: {e}")
        return None, None

    gripper_joint_index = -1
    grasp_timestep = find_grasp_timestep(episode, gripper_joint_index)

    if grasp_timestep is None:
        print("Could not identify a clear grasp event in the episode.")
        print("Consider using --manual-object-position or --skip-object-placement")
        return None, None

    print(f"Grasp event identified at timestep: {grasp_timestep}")

    grasp_qpos = episode["observation.state"][grasp_timestep]
    qpos_radians = np.deg2rad(grasp_qpos)
    robot_data.qpos[: len(qpos_radians)] = qpos_radians
    mujoco.mj_forward(robot_model, robot_data)

    try:
        gripper_site_name = "gripperframe"
        gripper_position = robot_data.site(gripper_site_name).xpos.copy()
        gripper_orientation_mat = robot_data.site(gripper_site_name).xmat.reshape(3, 3).copy()
        gripper_orientation_quat = np.empty(4)
        mujoco.mju_mat2Quat(gripper_orientation_quat, gripper_orientation_mat.flatten())
        print(f"Estimated grasped object position: {gripper_position}")
    except KeyError:
        print(f"Error: Site '{gripper_site_name}' not found in the robot XML.")
        return None, None

    return gripper_position, gripper_orientation_quat
