#!/usr/bin/env python3
"""
Analyze gripper geometry to determine jaw positions and center point.
"""
import mujoco
import numpy as np


def analyze_gripper_at_qpos(model, data, gripper_qpos_value):
    """
    Analyze gripper geometry at a specific gripper joint position.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        gripper_qpos_value: Value for the gripper joint (in radians)
    """
    # Set all joints to neutral (check actual qpos size)
    print(f"qpos size: {data.qpos.size}, nq: {model.nq}")
    neutral_pose = np.array([0.03755415, -1.7234037, 1.6718199, 1.2405578, -1.411793, 0.02459861])
    data.qpos[:len(neutral_pose)] = neutral_pose

    # Set gripper joint if it exists in this model
    gripper_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    if gripper_jnt_id != -1:
        gripper_qpos_addr = model.jnt_qposadr[gripper_jnt_id]
        data.qpos[gripper_qpos_addr] = gripper_qpos_value

    # Run forward kinematics
    mujoco.mj_forward(model, data)

    # Get gripperframe site position and orientation
    gripperframe_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    gripperframe_pos = data.site_xpos[gripperframe_site_id].copy()
    gripperframe_mat = data.site_xmat[gripperframe_site_id].reshape(3, 3).copy()

    # Get moving jaw body position
    moving_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    moving_jaw_pos = data.xpos[moving_jaw_id].copy()

    # Get gripper body position (fixed jaw reference)
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    gripper_body_pos = data.xpos[gripper_body_id].copy()

    print(f"\n{'='*80}")
    print(f"Gripper joint value: {gripper_qpos_value:.4f} rad ({np.rad2deg(gripper_qpos_value):.1f}°)")
    print(f"{'='*80}")
    print(f"\nGripperframe site (world):  {gripperframe_pos}")
    print(f"Gripper body (world):       {gripper_body_pos}")
    print(f"Moving jaw body (world):    {moving_jaw_pos}")

    # Calculate offset from gripperframe to gripper body
    offset_body = gripper_body_pos - gripperframe_pos
    print(f"\nOffset (gripper body - gripperframe): {offset_body}")

    # Calculate offset from gripperframe to moving jaw
    offset_moving = moving_jaw_pos - gripperframe_pos
    print(f"Offset (moving jaw - gripperframe):   {offset_moving}")

    # Transform world offsets to gripperframe local coordinates
    offset_body_local = gripperframe_mat.T @ offset_body
    offset_moving_local = gripperframe_mat.T @ offset_moving

    print(f"\nIn gripperframe local coordinates:")
    print(f"  Gripper body offset: {offset_body_local}")
    print(f"  Moving jaw offset:   {offset_moving_local}")

    # Estimate fixed jaw position (assuming it's part of gripper body)
    # For a parallel jaw gripper, fixed jaw might be symmetric to moving jaw
    # Let's check the gripper geoms
    print(f"\nGripper body geoms:")
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == gripper_body_id:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            geom_pos = model.geom_pos[geom_id]
            geom_type = model.geom_type[geom_id]
            print(f"  {geom_name}: type={geom_type}, pos={geom_pos}")

    print(f"\nMoving jaw geoms:")
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == moving_jaw_id:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            geom_pos = model.geom_pos[geom_id]
            geom_pos_world = data.geom_xpos[geom_id]
            geom_type = model.geom_type[geom_id]
            print(f"  {geom_name}: type={geom_type}, local_pos={geom_pos}, world_pos={geom_pos_world}")

    # Calculate jaw opening distance (approximate)
    jaw_opening = np.linalg.norm(moving_jaw_pos - gripper_body_pos)
    print(f"\nApproximate jaw opening: {jaw_opening*1000:.2f} mm")

    # Suggest center point between jaws (midpoint)
    center_between_jaws = (gripper_body_pos + moving_jaw_pos) / 2
    center_offset_from_gripperframe = center_between_jaws - gripperframe_pos
    center_offset_local = gripperframe_mat.T @ center_offset_from_gripperframe

    print(f"\nEstimated center between jaws (world): {center_between_jaws}")
    print(f"Offset from gripperframe (world):      {center_offset_from_gripperframe}")
    print(f"Offset from gripperframe (local):      {center_offset_local}")

    # Print gripperframe orientation
    print(f"\nGripperframe orientation matrix:")
    print(gripperframe_mat)
    print(f"X-axis (gripper local): {gripperframe_mat[:, 0]}")
    print(f"Y-axis (gripper local): {gripperframe_mat[:, 1]}")
    print(f"Z-axis (gripper local): {gripperframe_mat[:, 2]}")

    return center_offset_local


def main():
    # Load robot model
    model_path = "so101-assets/so101_new_calib.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Analyze at different gripper positions
    print("\n" + "="*80)
    print("GRIPPER GEOMETRY ANALYSIS")
    print("="*80)

    # Closed gripper (near min range: -0.174 rad ≈ -10°)
    offset_closed = analyze_gripper_at_qpos(model, data, -0.1)

    # Half-open gripper (mid range)
    offset_half = analyze_gripper_at_qpos(model, data, 0.785)  # 45°

    # Open gripper (near max range: 1.745 rad ≈ 100°)
    offset_open = analyze_gripper_at_qpos(model, data, 1.5)

    # Average offset (should be relatively constant for parallel jaws)
    avg_offset = (offset_closed + offset_half + offset_open) / 3
    print(f"\n{'='*80}")
    print(f"SUMMARY: Average offset from gripperframe to jaw center (local coords)")
    print(f"{'='*80}")
    print(f"Average offset: {avg_offset}")
    print(f"\nRecommended offset to use in code:")
    print(f"GRIPPER_CENTER_OFFSET = np.array([{avg_offset[0]:.6f}, {avg_offset[1]:.6f}, {avg_offset[2]:.6f}])")


if __name__ == "__main__":
    main()
