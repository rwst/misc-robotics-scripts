# Gripper Calibration and Object Placement

## Overview

This document explains how the gripper geometry is calibrated and how objects are placed at the grasp location.

## Gripper Reference Frame

The SO-ARM100 gripper has a reference frame called `gripperframe` defined as a MuJoCo site:

```xml
<site group="3" name="gripperframe" pos="-0.0079 -0.000218121 -0.0981274" quat="0.707107 -0 0.707107 -2.37788e-17"/>
```

This site is positioned relative to the gripper body at:
- **Position**: (-7.9mm, -0.2mm, -98.1mm) from gripper body origin
- **Purpose**: End-effector tracking and forward kinematics reference point

## Gripper Coordinate System

The gripperframe has the following local axes:
- **X-axis**: Along gripper longitudinal axis (points roughly forward/down)
- **Y-axis**: Jaw opening direction (perpendicular to jaws when closed)
- **Z-axis**: Perpendicular to both (cross product of X and Y)

## Jaw Center Offset

The `gripperframe` site is **not** located at the center between the gripper jaws. Through geometric analysis (see `analyze_gripper_geometry.py`), we determined the offset from `gripperframe` to the actual jaw center:

```python
GRIPPER_CENTER_OFFSET = np.array([-0.0864274, 0.00961812, 0.018])
```

In gripperframe local coordinates:
- **X**: -86.4mm (jaw center is behind the gripperframe reference)
- **Y**: +9.6mm (slight lateral offset)
- **Z**: +18.0mm (vertical offset)
- **Total magnitude**: 88.8mm

## Object Placement Algorithm

When placing a grasped object in the scene (`detect_grasp_and_compute_object_pose`):

### 1. Grasp Detection
- Analyzes gripper joint velocity to find the grasp moment
- Identifies timestep where gripper closes and stabilizes

### 2. Forward Kinematics
- Sets robot to grasp configuration using `qpos` from dataset
- Runs `mj_forward` to compute gripperframe world position and orientation

### 3. Offset Transformation
```python
# Transform offset from gripper local frame to world frame
gripper_center_offset_world = gripperframe_orientation_mat @ GRIPPER_CENTER_OFFSET

# Compute object position at jaw center
object_position = gripperframe_position + gripper_center_offset_world
```

### 4. Orientation Alignment
The object's long axis (Y-axis, 30mm for object1) is oriented **perpendicular** to the jaw opening direction:

```python
# Gripper frame: X=forward, Y=opening, Z=perpendicular
# Object frame: X=7mm, Y=15mm (long), Z=5mm
# Desired: object Y-axis ⊥ gripper Y-axis

object_orientation_mat[:, 0] = gripperframe_orientation_mat[:, 1]  # object X = gripper Y
object_orientation_mat[:, 1] = gripperframe_orientation_mat[:, 2]  # object Y = gripper Z
object_orientation_mat[:, 2] = gripperframe_orientation_mat[:, 0]  # object Z = gripper X
```

This ensures the object is grasped along its short dimension (jaws close on the 14mm side, not the 30mm side).

### 5. Z-Height Correction
Finally, `place_object_in_scene` in `environment.py` overrides the Z-coordinate to ensure the object's bottom surface sits on the ground:

```python
computed_z_height = compute_object_z_height(model, object_name)
object_position[2] = computed_z_height  # Replaces Z with geometry-based height
```

## Object Geometry

From `so101_with_objects.xml`:
```xml
<body name="object1" pos="0.15 -0.15 0.025">
  <geom type="box" size="0.007 0.015 0.005" ... />
</body>
```

- **Half-sizes**: 7mm × 15mm × 5mm (MuJoCo uses half-extents)
- **Full dimensions**: 14mm × 30mm × 10mm
- **Reference point**: Geometric center (standard for MuJoCo bodies)
- **Long axis**: Y-axis (30mm)

## Verification

Run `analyze_gripper_geometry.py` to verify the calibration:
```bash
python mujoco-so101/rd/analyze_gripper_geometry.py
```

This analyzes the gripper at different joint positions and computes the average offset from gripperframe to jaw center.

## References

- MuJoCo geometry conventions: https://mujoco.readthedocs.io/en/stable/modeling.html
- SO-ARM100 repository: https://github.com/TheRobotStudio/SO-ARM100
- Gripper specifications: Max opening ~85mm (joint range: -10° to 100°)
