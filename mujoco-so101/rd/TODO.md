# TODO: replay_dataset.py Issues and Improvements

Generated: 2025-12-23
Analysis of: `mujoco-so101/rd/replay_dataset.py`

## Critical Issues

### 1. Hardcoded Object Z-Height
**Lines: 334, 387**
- Z-height is hardcoded to 0.025m for all objects
- Uses `args.object_name` but assumes all objects have the same height
- **Fix**: Add `--object-height` argument or infer from XML model
- **Impact**: Incorrect object placement for objects with different heights

### 2. Action Scaling Assumption
**Line: 452**
- Assumes all LeRobot dataset actions are in degrees: `scaled_action = np.deg2rad(action)`
- No validation that this assumption holds for the specific dataset
- **Fix**: Add metadata check or explicit action-unit specification argument
- **Impact**: Could cause completely wrong actions if assumption is violated

### 3. Grasp Detection Gripper Direction Assumption
**Lines: 81-85**
- Assumes gripper closing always produces negative velocity (`np.argmin`)
- May fail if gripper joint convention is reversed
- **Fix**: Make gripper closing direction configurable or auto-detect
- **Impact**: Fails to detect grasp events on robots with opposite joint conventions

## Bugs

### 5. Resource Leak on Early Return
**Lines: Throughout main()**
- Multiple early returns (lines 198, 201, 215, 225, 240, 251, 271, 279, 298, 428)
- Some paths may not close resources (env, FK models)
- **Fix**: Use try-finally or context managers to ensure cleanup
- **Impact**: File handles or memory leaks in error paths

### 6. Duplicate FK Model Loading
**Lines: 267, 435**
- FK model loaded twice for grasp detection and state comparison
- Inefficient, wastes memory and load time
- **Fix**: Load once and reuse if both features needed
- **Impact**: Minor performance issue, ~2x load time for FK model

### 7. Camera Warning Suppression
**Lines: 319-321**
- Checks if `camera_id != -1` but doesn't warn user if camera not found
- Silent failure to adjust camera position
- **Fix**: Print warning if camera doesn't exist
- **Impact**: User unaware that camera adjustment failed

### 8. Commented Code Suggests Position Bug
**Line: 291**
- `#gripper_position = gripper_position - 0.05` commented out
- Suggests there may be a position offset issue that was being debugged
- **Fix**: Remove dead code or document why offset was considered/rejected
- **Impact**: Code clarity, possible unresolved offset issue

## Design Issues

### 10. Code Duplication: Object Placement
**Lines: 331-343 and 385-395**
- Object placement logic duplicated for start-image-only vs video modes
- **Fix**: Extract into `place_object(env, qpos_addr, gripper_position, gripper_orientation_quat, object_name)` function
- **Impact**: Code maintainability, risk of inconsistency

### 11. Magic Numbers Throughout
**Lines: Various**
- `500` (max_steps_per_move) - line 399
- `1e-3` (movement_epsilon) - line 400
- `0.3` (camera offset) - line 321
- `0.025` (object z-height) - lines 334, 387
- `1e-3` (grasp detection tolerance) - line 84
- **Fix**: Define as named constants at module level or make configurable
- **Impact**: Code readability and maintainability

## Missing Features / Enhancements

### 13. Dataset Convention Not Verified
**Lines: 449-504**
- Assumes state[i+1] is result of action[i]
- No verification this matches the actual dataset convention
- **Fix**: Add assertion or documentation of convention, verify with dataset metadata
- **Impact**: Incorrect state comparisons if convention differs

### 14. Fixed-Step Mode Has No Validation
**Lines: 454-467**
- Fixed-step mode doesn't check if movement actually completed
- No way to know if N steps was sufficient
- **Fix**: Optionally report final position change norm as warning if large
- **Impact**: User unaware if fixed steps insufficient

### 15. Warning Still Prints When Object Placement Skipped
**Lines: 340-343, 394-395**
- Prints "Object found but no position available" even with `--skip-object-placement`
- **Fix**: Suppress warning when explicitly skipping placement
- **Impact**: Confusing output for users who intentionally skip placement

### 16. No Gripper Orientation Validation
**Lines: 256, 294**
- Quaternion computed from rotation matrix but never validated (unit norm, etc.)
- Could silently fail with invalid quaternion
- **Fix**: Assert quaternion norm is close to 1.0
- **Impact**: Silent failures in object orientation

### 17. State Comparison Skips Last Action
**Line: 498**
- Checks `i + 1 < len(episode["observation.state"])` which skips comparing last action's result
- **Fix**: This is actually correct (no next state for last action), but could document why
- **Impact**: User confusion about why last action isn't compared

## Code Quality

### 19. Inconsistent Error Messages
**Lines: Throughout**
- Some errors use f-strings, some use concatenation
- Some print to stdout, should use stderr for errors
- **Fix**: Standardize error formatting, use `sys.stderr` or logging module
- **Impact**: Code consistency

### 20. No Logging Framework
**Lines: Throughout**
- Uses print() statements instead of proper logging
- No log levels (debug, info, warning, error)
- **Fix**: Use Python `logging` module instead of print
- **Impact**: Difficult to control output verbosity granularly

### 21. Verbosity Flag Incomplete
**Lines: 457, 473, 588**
- `--verbosity` flag only affects action execution progress
- Warnings and other output still printed regardless of verbosity
- **Fix**: Apply verbosity consistently to all non-error output
- **Impact**: Cannot achieve truly quiet execution

## Performance

### 22. Unnecessary Data Copies
**Lines: 77, 210, 243, 245**
- `np.array()` called on data that may already be arrays
- **Fix**: Use `np.asarray()` instead (no-copy if already array)
- **Impact**: Minor memory/performance improvement

### 23. State Comparison Could Be Vectorized
**Lines: 573-585**
- Per-timestep state error computation in Python loop
- **Fix**: Compute all errors at once with vectorized operations after replay
- **Impact**: Faster state comparison for large episodes

## Documentation

### 24. Missing Docstrings
**Lines: 15-70**
- SO101Env class methods lack docstrings (except class itself)
- **Fix**: Add docstrings to `__init__`, `_get_obs`, `reset_model`, `step`

### 25. Unclear Grasp Detection Algorithm
**Lines: 73-87**
- `find_grasp_timestep` docstring doesn't explain algorithm
- **Fix**: Document: "Finds grasp by detecting peak gripper closing velocity followed by stabilization"

### 26. No Example Usage in Docstring
**Lines: 90-93**
- Main function has brief description but no usage examples
- **Fix**: Add examples section showing common use cases

## Testing Suggestions

### 27. No Unit Tests
- No test coverage for grasp detection, FK computation, state comparison
- **Recommendation**: Add tests for:
  - `find_grasp_timestep()` with synthetic data
  - Object placement calculation
  - State comparison metrics
  - Edge cases (empty episode, no grasp, etc.)

### 28. No Integration Tests
- No end-to-end test with known dataset
- **Recommendation**: Add test that replays a small known episode and validates output

## Priority Ranking

**P0 (Critical):** 1, 2, 4
**P1 (High):** 3, 5, 12, 13, 18
**P2 (Medium):** 6, 7, 8, 9, 10, 14, 15, 16
**P3 (Low):** 11, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28

## Recommended Refactoring Approach

3. Add unit tests (Issue #27) - validates fixes
4. Fix critical assumptions (Issues #1, #2, #3) - prevents wrong behavior
5. Add proper error handling (Issue #5) - prevents resource leaks
6. Address code quality issues (Issues #10, #11, #19, #20, #21)
7. Add missing features (Issues #13, #14, #15, #16)
8. Performance optimizations (Issues #6, #22, #23)
9. Documentation improvements (Issues #24, #25, #26)
