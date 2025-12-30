#!/usr/bin/env python3
"""
Unit tests for replay dataset modules.

These tests cover functions that can be tested without requiring special datasets or MuJoCo models.
"""

import unittest
import numpy as np
from argparse import Namespace
from unittest.mock import Mock, patch
import io
import sys

# Import modules to test
from data_loader import validate_input_args
from grasp_detection import find_grasp_timestep
from state_comparison import validate_episode_data, print_state_comparison_summary
from environment import compute_object_z_height


class TestDataLoader(unittest.TestCase):
    """Tests for data_loader.py functions."""

    def test_validate_input_args_both_sources(self):
        """Test that validation fails when both npy path and dataset options are provided."""
        args = Namespace(
            actions_npy_path="actions.npy",
            repo_id="some/dataset",
            episode_index=0
        )
        result = validate_input_args(args)
        self.assertFalse(result)

    def test_validate_input_args_no_sources(self):
        """Test that validation fails when no data source is provided."""
        args = Namespace(
            actions_npy_path=None,
            repo_id=None,
            episode_index=None
        )
        result = validate_input_args(args)
        self.assertFalse(result)

    def test_validate_input_args_npy_only(self):
        """Test that validation succeeds with only npy path."""
        args = Namespace(
            actions_npy_path="actions.npy",
            repo_id=None,
            episode_index=None
        )
        result = validate_input_args(args)
        self.assertTrue(result)

    def test_validate_input_args_dataset_complete(self):
        """Test that validation succeeds with complete dataset info."""
        args = Namespace(
            actions_npy_path=None,
            repo_id="some/dataset",
            episode_index=0
        )
        result = validate_input_args(args)
        self.assertTrue(result)

    def test_validate_input_args_dataset_incomplete_no_index(self):
        """Test that validation fails with repo_id but no episode_index."""
        args = Namespace(
            actions_npy_path=None,
            repo_id="some/dataset",
            episode_index=None
        )
        result = validate_input_args(args)
        self.assertFalse(result)

    def test_validate_input_args_dataset_incomplete_no_repo(self):
        """Test that validation fails with episode_index but no repo_id."""
        args = Namespace(
            actions_npy_path=None,
            repo_id=None,
            episode_index=0
        )
        result = validate_input_args(args)
        self.assertFalse(result)


class TestGraspDetection(unittest.TestCase):
    """Tests for grasp_detection.py functions."""

    def test_find_grasp_timestep_clear_grasp(self):
        """Test grasp detection with clear grasp event."""
        # Create synthetic data: gripper closes then stabilizes
        qpos = np.array([
            [0.0],  # Initial open position
            [0.0],  # Still open
            [-5.0], # Start closing (negative velocity)
            [-10.0], # Peak closing velocity
            [-12.0], # Slowing down
            [-13.0], # Almost stopped
            [-13.0], # Stabilized (velocity ~0)
            [-13.0], # Still stable
        ])
        episode = {"observation.state": qpos}

        result = find_grasp_timestep(episode, 0)

        # Should detect stabilization at timestep 6
        self.assertEqual(result, 6)

    def test_find_grasp_timestep_no_stabilization(self):
        """Test grasp detection when gripper never stabilizes."""
        # Gripper keeps moving
        qpos = np.array([
            [0.0],
            [-5.0],
            [-10.0],
            [-15.0],
            [-20.0],
            [-25.0],
        ])
        episode = {"observation.state": qpos}

        result = find_grasp_timestep(episode, 0)

        # Should return None if no stabilization detected
        self.assertIsNone(result)

    def test_find_grasp_timestep_immediate_stabilization(self):
        """Test grasp detection with immediate stabilization."""
        # Gripper closes and immediately stabilizes
        qpos = np.array([
            [0.0],
            [-10.0], # Large jump (peak velocity)
            [-10.0], # Immediate stabilization
            [-10.0],
        ])
        episode = {"observation.state": qpos}

        result = find_grasp_timestep(episode, 0)

        # Should detect stabilization at timestep 2
        self.assertEqual(result, 2)

    def test_find_grasp_timestep_multiple_joints(self):
        """Test grasp detection with multiple joints (only last is gripper)."""
        # 3 joints, gripper is last column
        qpos = np.array([
            [10.0, 20.0, 0.0],
            [10.5, 21.0, 0.0],
            [11.0, 22.0, -5.0],
            [11.5, 23.0, -10.0],
            [12.0, 24.0, -12.0],
            [12.5, 25.0, -12.0],  # Gripper stabilized
        ])
        episode = {"observation.state": qpos}

        result = find_grasp_timestep(episode, -1)  # -1 = last joint

        self.assertEqual(result, 5)


class TestStateComparison(unittest.TestCase):
    """Tests for state_comparison.py functions."""

    def setUp(self):
        """Set up mock environment for tests."""
        self.mock_env = Mock()
        self.mock_env.action_space.shape = [6]  # 6 DOF robot

    def test_validate_episode_data_valid(self):
        """Test validation with valid episode data."""
        episode = {
            "action": np.random.randn(10, 6),
            "observation.state": np.random.randn(11, 6)
        }

        actions, num_joints, valid = validate_episode_data(episode, self.mock_env)

        self.assertTrue(valid)
        self.assertEqual(num_joints, 6)
        self.assertEqual(actions.shape, (10, 6))

    def test_validate_episode_data_no_actions(self):
        """Test validation fails with no actions."""
        episode = {
            "action": None,
            "observation.state": None
        }

        actions, num_joints, valid = validate_episode_data(episode, self.mock_env)

        self.assertFalse(valid)
        self.assertIsNone(actions)
        self.assertIsNone(num_joints)

    def test_validate_episode_data_empty_actions(self):
        """Test validation fails with empty actions array."""
        episode = {
            "action": np.array([]),
            "observation.state": None
        }

        actions, num_joints, valid = validate_episode_data(episode, self.mock_env)

        self.assertFalse(valid)

    def test_validate_episode_data_wrong_action_dim(self):
        """Test validation fails when action dimension doesn't match environment."""
        episode = {
            "action": np.random.randn(10, 7),  # 7 instead of 6
            "observation.state": None
        }

        actions, num_joints, valid = validate_episode_data(episode, self.mock_env)

        self.assertFalse(valid)

    def test_validate_episode_data_1d_actions(self):
        """Test validation fails with 1D actions array."""
        episode = {
            "action": np.random.randn(10),  # Should be 2D
            "observation.state": None
        }

        actions, num_joints, valid = validate_episode_data(episode, self.mock_env)

        self.assertFalse(valid)

    def test_validate_episode_data_insufficient_states(self):
        """Test validation fails when states < actions."""
        episode = {
            "action": np.random.randn(10, 6),
            "observation.state": np.random.randn(5, 6)  # Not enough states
        }

        actions, num_joints, valid = validate_episode_data(episode, self.mock_env)

        self.assertFalse(valid)

    def test_validate_episode_data_1d_states(self):
        """Test validation fails with 1D states array."""
        episode = {
            "action": np.random.randn(10, 6),
            "observation.state": np.random.randn(10)  # Should be 2D
        }

        actions, num_joints, valid = validate_episode_data(episode, self.mock_env)

        self.assertFalse(valid)

    def test_validate_episode_data_no_states_ok(self):
        """Test validation succeeds when states are None (optional)."""
        episode = {
            "action": np.random.randn(10, 6),
            "observation.state": None
        }

        actions, num_joints, valid = validate_episode_data(episode, self.mock_env)

        self.assertTrue(valid)
        self.assertEqual(actions.shape, (10, 6))

    def test_print_state_comparison_summary_empty(self):
        """Test that summary handles empty state_errors gracefully."""
        state_errors = []

        # Should not crash, just return early
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            print_state_comparison_summary(state_errors, 6)
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Should output nothing for empty errors
        self.assertEqual(output, "")

    def test_print_state_comparison_summary_with_data(self):
        """Test that summary prints statistics correctly."""
        # Create synthetic error data
        state_errors = [
            {
                'timestep': 0,
                'mae': 1.0,
                'rmse': 1.2,
                'max_error': 2.0,
                'per_joint_errors': np.array([0.5, 1.0, 1.5, 2.0, 0.8, 1.2])
            },
            {
                'timestep': 1,
                'mae': 1.5,
                'rmse': 1.8,
                'max_error': 2.5,
                'per_joint_errors': np.array([0.8, 1.2, 1.8, 2.5, 1.0, 1.5])
            },
        ]

        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            print_state_comparison_summary(state_errors, 6)
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__

        # Check that output contains expected sections
        self.assertIn("STATE COMPARISON SUMMARY", output)
        self.assertIn("Mean Absolute Error (MAE)", output)
        self.assertIn("Root Mean Square Error (RMSE)", output)
        self.assertIn("Per-joint error statistics", output)

        # Check some specific values
        self.assertIn("1.2500", output)  # Mean of MAE (1.0 + 1.5) / 2


class TestEnvironment(unittest.TestCase):
    """Tests for environment.py functions."""

    def test_compute_object_z_height_box(self):
        """Test z-height computation for box geometry."""
        # Create mock model for a box (2cm x 1cm x 0.5cm, half-heights)
        mock_model = Mock()
        mock_model.ngeom = 1

        # Mock joint
        mock_model.jnt_bodyid = [5]  # Joint attached to body 5

        # Mock geom attached to body 5
        mock_model.geom_bodyid = [5]
        mock_model.geom_type = [6]  # mjGEOM_BOX = 6
        mock_model.geom_size = [np.array([0.01, 0.005, 0.0025])]  # Half-sizes
        mock_model.geom_pos = [np.array([0.0, 0.0, 0.0])]  # Centered on body

        # Mock mj_name2id to return joint_id=0
        with patch('mujoco.mj_name2id', return_value=0):
            z_height = compute_object_z_height(mock_model, "test_object")

        # For box, z_height should be size[2] = 0.0025m (2.5mm)
        self.assertAlmostEqual(z_height, 0.0025, places=6)

    def test_compute_object_z_height_sphere(self):
        """Test z-height computation for sphere geometry."""
        mock_model = Mock()
        mock_model.ngeom = 1
        mock_model.jnt_bodyid = [3]
        mock_model.geom_bodyid = [3]
        mock_model.geom_type = [2]  # mjGEOM_SPHERE = 2
        mock_model.geom_size = [np.array([0.05, 0.0, 0.0])]  # Radius 5cm
        mock_model.geom_pos = [np.array([0.0, 0.0, 0.0])]

        with patch('mujoco.mj_name2id', return_value=0):
            z_height = compute_object_z_height(mock_model, "sphere_object")

        # For sphere, z_height should be radius = 0.05m (5cm)
        self.assertAlmostEqual(z_height, 0.05, places=6)

    def test_compute_object_z_height_cylinder(self):
        """Test z-height computation for cylinder geometry."""
        mock_model = Mock()
        mock_model.ngeom = 1
        mock_model.jnt_bodyid = [2]
        mock_model.geom_bodyid = [2]
        mock_model.geom_type = [5]  # mjGEOM_CYLINDER = 5
        mock_model.geom_size = [np.array([0.02, 0.03, 0.0])]  # Radius 2cm, half-height 3cm
        mock_model.geom_pos = [np.array([0.0, 0.0, 0.0])]

        with patch('mujoco.mj_name2id', return_value=0):
            z_height = compute_object_z_height(mock_model, "cylinder_object")

        # For cylinder, z_height should be size[1] = 0.03m (3cm half-height)
        self.assertAlmostEqual(z_height, 0.03, places=6)

    def test_compute_object_z_height_capsule(self):
        """Test z-height computation for capsule geometry."""
        mock_model = Mock()
        mock_model.ngeom = 1
        mock_model.jnt_bodyid = [4]
        mock_model.geom_bodyid = [4]
        mock_model.geom_type = [3]  # mjGEOM_CAPSULE = 3
        mock_model.geom_size = [np.array([0.01, 0.02, 0.0])]  # Radius 1cm, half-height 2cm
        mock_model.geom_pos = [np.array([0.0, 0.0, 0.0])]

        with patch('mujoco.mj_name2id', return_value=0):
            z_height = compute_object_z_height(mock_model, "capsule_object")

        # For capsule, z_height should be size[1] + size[0] = 0.02 + 0.01 = 0.03m
        self.assertAlmostEqual(z_height, 0.03, places=6)

    def test_compute_object_z_height_with_offset(self):
        """Test z-height computation when geom has position offset."""
        mock_model = Mock()
        mock_model.ngeom = 1
        mock_model.jnt_bodyid = [1]
        mock_model.geom_bodyid = [1]
        mock_model.geom_type = [6]  # mjGEOM_BOX = 6
        mock_model.geom_size = [np.array([0.01, 0.01, 0.01])]  # 1cm half-height
        # Geom is offset 0.5cm above body origin
        mock_model.geom_pos = [np.array([0.0, 0.0, 0.005])]

        with patch('mujoco.mj_name2id', return_value=0):
            z_height = compute_object_z_height(mock_model, "offset_object")

        # z_height = size[2] - pos[2] = 0.01 - 0.005 = 0.005m
        self.assertAlmostEqual(z_height, 0.005, places=6)

    def test_compute_object_z_height_multiple_geoms(self):
        """Test z-height computation with multiple geoms (takes minimum)."""
        mock_model = Mock()
        mock_model.ngeom = 2
        mock_model.jnt_bodyid = [7]
        mock_model.geom_bodyid = [7, 7]  # Both geoms on same body
        mock_model.geom_type = [6, 6]  # Two boxes (mjGEOM_BOX = 6)
        # First geom: larger (3cm half-height)
        # Second geom: smaller (1cm half-height) - this should be used
        mock_model.geom_size = [
            np.array([0.02, 0.02, 0.03]),
            np.array([0.01, 0.01, 0.01])
        ]
        mock_model.geom_pos = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        ]

        with patch('mujoco.mj_name2id', return_value=0):
            z_height = compute_object_z_height(mock_model, "multi_geom_object")

        # Should use minimum: 0.01m from second geom
        self.assertAlmostEqual(z_height, 0.01, places=6)

    def test_compute_object_z_height_invalid_object(self):
        """Test z-height computation with invalid object name."""
        mock_model = Mock()

        # mj_name2id returns -1 for invalid object
        with patch('mujoco.mj_name2id', return_value=-1):
            z_height = compute_object_z_height(mock_model, "nonexistent_object")

        # Should return None for invalid object
        self.assertIsNone(z_height)

    def test_compute_object_z_height_no_geoms(self):
        """Test z-height computation when body has no geoms."""
        mock_model = Mock()
        mock_model.ngeom = 2
        mock_model.jnt_bodyid = [5]
        # Geoms belong to different bodies (not body 5)
        mock_model.geom_bodyid = [1, 2]

        with patch('mujoco.mj_name2id', return_value=0):
            z_height = compute_object_z_height(mock_model, "no_geom_object")

        # Should return default fallback (0.025m)
        self.assertAlmostEqual(z_height, 0.025, places=6)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple modules."""

    def test_grasp_detection_and_validation_flow(self):
        """Test a typical flow from grasp detection to validation."""
        # Create synthetic episode with grasp
        qpos = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 10.0, 15.0, 20.0, 25.0, 0.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, -5.0],
            [15.0, 30.0, 45.0, 60.0, 75.0, -10.0],
            [20.0, 40.0, 60.0, 80.0, 100.0, -10.0],  # Gripper stabilized
        ])

        actions = np.random.randn(4, 6)

        episode = {
            "observation.state": qpos,
            "action": actions
        }

        # Test grasp detection
        grasp_timestep = find_grasp_timestep(episode, -1)
        self.assertEqual(grasp_timestep, 4)

        # Test validation
        mock_env = Mock()
        mock_env.action_space.shape = [6]

        actions_out, num_joints, valid = validate_episode_data(episode, mock_env)
        self.assertTrue(valid)
        self.assertEqual(num_joints, 6)

    def test_edge_case_single_action(self):
        """Test edge case with single action."""
        episode = {
            "action": np.random.randn(1, 6),
            "observation.state": np.random.randn(2, 6)
        }

        mock_env = Mock()
        mock_env.action_space.shape = [6]

        actions, num_joints, valid = validate_episode_data(episode, mock_env)

        self.assertTrue(valid)
        self.assertEqual(actions.shape, (1, 6))


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestGraspDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestStateComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
