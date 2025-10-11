# misc-robotics-scripts

## Folder `so101-assets`
* The STL files are from https://huggingface.co/haixuantao/dora-bambot/tree/main/URDF/assets.
* The URDF and XML are from https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

## Folder `mujoco-so101`
* `so101-manual.py`: short manual action as proof of concept

## Scripts

### `inspect_npy.py`
Wrapper around `numpy.load()`

### `hf_get_dataset.py`
Wrapper around HF's `snapshot_download()`
 
### `infer_single_step_act.py`
Single-step an ACT model with random input

```
python infer_single_step_act.py --policy_path funXedu/so101_act_lego_brick_v2
Loading policy from: funXedu/so101_act_lego_brick_v2
WARNING:root:No accelerated backend detected. Using default cpu, this will be slow.
WARNING:root:Device 'cuda' is not available. Switching to 'cpu'.
WARNING:root:No accelerated backend detected. Using default cpu, this will be slow.
WARNING:root:Device 'cuda' is not available. Switching to 'cpu'.
Inferred state dimension from policy config: 6
Running inference with dummy input shapes:
  observation.images.front: (1, 3, 128, 128)
  observation.images.gripperR: (1, 3, 128, 128)
  observation.state: (1, 6)
tensor([[[[0.9872, 0.3257, 0.2866,  ..., 0.3699, 0.7518, 0.6864],
          [0.3430, 0.2379, 0.4583,  ..., 0.7107, 0.0732, 0.5239],
          [0.9494, 0.1332, 0.6202,  ..., 0.3370, 0.3599, 0.0659],
          ...,
          [0.2287, 0.9007, 0.7030,  ..., 0.8669, 0.2872, 0.3383],
          [0.1019, 0.9628, 0.6114,  ..., 0.0046, 0.5724, 0.9564],
          [0.0780, 0.0663, 0.3544,  ..., 0.5479, 0.1420, 0.3933]],

         [[0.1338, 0.2799, 0.8795,  ..., 0.7126, 0.5827, 0.9233],
          [0.4888, 0.7705, 0.4002,  ..., 0.8675, 0.9421, 0.7290],
          [0.4451, 0.8599, 0.7283,  ..., 0.3589, 0.5292, 0.7011],
          ...,
          [0.2664, 0.7879, 0.6692,  ..., 0.1078, 0.3471, 0.5821],
          [0.9395, 0.2547, 0.6226,  ..., 0.8168, 0.2632, 0.9664],
          [0.8507, 0.3299, 0.4892,  ..., 0.7074, 0.8546, 0.1828]],

         [[0.8977, 0.9553, 0.5894,  ..., 0.1173, 0.8553, 0.4038],
          [0.3746, 0.3315, 0.9646,  ..., 0.9830, 0.7453, 0.4405],
          [0.7416, 0.6001, 0.5324,  ..., 0.4933, 0.4217, 0.2348],
          ...,
          [0.4054, 0.6204, 0.3100,  ..., 0.0749, 0.5280, 0.9882],
          [0.6373, 0.3386, 0.6455,  ..., 0.1144, 0.8452, 0.4422],
          [0.8866, 0.0161, 0.8327,  ..., 0.6278, 0.9632, 0.7691]]]])

--- Inference Result ---
Inference Time: 41.23 ms
Predicted Action Type: <class 'torch.Tensor'>
Predicted Action Shape: (1, 6)
Predicted Action: [0.16167816519737244, 0.17730316519737244, -0.19864612817764282, 0.3916305899620056, 0.5123617053031921, 0.7893270254135132]
------------------------
Single-step inference complete.
```


### `infer_single_step_smolvla.py`
Single-step an ACT model with random input

```
python infer_single_step_smolvla.py 
Loading policy from: lerobot/smolvla_base
WARNING:root:No accelerated backend detected. Using default cpu, this will be slow.
WARNING:root:Device 'cuda' is not available. Switching to 'cpu'.
WARNING:root:No accelerated backend detected. Using default cpu, this will be slow.
WARNING:root:Device 'cuda' is not available. Switching to 'cpu'.
Inferred state dimension from policy config: 6
Loading  HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
Reducing the number of VLM layers to 16 ...
Running inference with dummy input data...

--- Inference Result ---
Inference Time: 18586.25 ms
Predicted Action Type: <class 'torch.Tensor'>
Predicted Action Shape: (1, 6)
Predicted Action: [0.9534054398536682, 0.4724017381668091, 0.4320472478866577, 0.7170732021331787, 0.5062081813812256, 0.1788467913866043]
------------------------
Single-step inference complete.
```
