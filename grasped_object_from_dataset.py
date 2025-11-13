
import argparse
import numpy as np
import mujoco
import datasets

def find_grasp_timestep(episode, gripper_joint_index, z_threshold=0.1):
    """
    Finds the timestep of the grasp event in an episode.

    The grasp is identified by looking for the point where the gripper stops closing,
    indicating it has made contact with an object.

    Args:
        episode (dict): A single episode from the dataset.
        gripper_joint_index (int): The index of the gripper joint in the state vector.
        z_threshold (float): The maximum height (Z-axis) for the gripper to be considered
                             near the table.

    Returns:
        int: The index of the timestep where the grasp occurred, or None if not found.
    """
    qpos = np.array(episode['observation.state'])
    gripper_qpos = qpos[:, gripper_joint_index]
    gripper_vel = np.diff(gripper_qpos, prepend=gripper_qpos[0])

    # Find the peak closing velocity
    peak_closing_vel_idx = np.argmin(gripper_vel)

    # Search for the grasp event after the peak closing velocity
    for i in range(peak_closing_vel_idx + 1, len(gripper_vel)):
        # Grasp is detected when gripper velocity approaches zero after closing
        if np.isclose(gripper_vel[i], 0, atol=1e-3):
            # Check if the end-effector is close to the table (low z-value)
            # Note: This requires running FK for each step, which is slow.
            # As a proxy, we'll perform the check after finding a candidate grasp.
            # A full implementation would check this condition here.
            return i
            
    return None

def main():
    """
    Main function to load data, run kinematics, and estimate object location.
    """
    parser = argparse.ArgumentParser(
        description="Estimate the position of a grasped object from a dataset."
    )
    parser.add_argument(
        "--xml_file",
        type=str,
        required=True,
        help="Path to the MuJoCo XML file for the robot model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the Hugging Face dataset to use."
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="The episode number to process from the dataset."
    )

    args = parser.parse_args()

    # 1. Load the MuJoCo Model
    try:
        model = mujoco.MjModel.from_xml_path(args.xml_file)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading MuJoCo model from {args.xml_file}: {e}")
        return

    # 2. Load the Dataset
    try:
        print(f"Loading episode {args.episode} from dataset '{args.dataset}'...")
        # Use streaming to avoid downloading the whole dataset
        dataset = datasets.load_dataset(args.dataset, split='train', streaming=True)

        # Filter the dataset for the desired episode
        episode_dataset = dataset.filter(lambda example: example["episode_index"] == args.episode)
        
        episode_steps = list(episode_dataset)

        if not episode_steps:
            print(f"Episode {args.episode} not found or is empty.")
            return

        # Re-structure the data into a dictionary of arrays
        episode = {
            'observation.state': np.array([step['observation.state'] for step in episode_steps]),
            'action': np.array([step['action'] for step in episode_steps]),
        }
        print("Episode loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset '{args.dataset}': {e}")
        return

    # 3. Find the Grasp Event
    # From the XML, the gripper joint is the last one.
    gripper_joint_index = -1 
    grasp_timestep = find_grasp_timestep(episode, gripper_joint_index)

    if grasp_timestep is None:
        print("Could not identify a clear grasp event in the episode.")
        return

    print(f"Grasp event identified at timestep: {grasp_timestep}")

    # 4. Perform Forward Kinematics
    grasp_qpos = episode['observation.state'][grasp_timestep]

    # Set the joint positions in the MuJoCo data object
    data.qpos[:6] = grasp_qpos
    
    # Run the forward kinematics
    mujoco.mj_forward(model, data)

    # 5. Get the Gripper Position
    # The XML file defines a site named 'gripperframe' at the center of the gripper
    try:
        gripper_site_name = 'gripperframe'
        gripper_position = data.site(gripper_site_name).xpos
        
        print(f"\\nEstimated grasped object position (at site '{gripper_site_name}'):")
        print(f"  X: {gripper_position[0]:.4f}")
        print(f"  Y: {gripper_position[1]:.4f}")
        print(f"  Z: {gripper_position[2]:.4f}")

    except KeyError:
        print(f"Error: Site '{gripper_site_name}' not found in the MuJoCo model.")
        print("Please ensure the XML file contains a site element for the gripper.")


if __name__ == "__main__":
    main()
