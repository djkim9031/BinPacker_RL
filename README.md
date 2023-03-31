# 3D BinPacker_RL
3D Bin Packing with Reinforcement Learning Agent and NVIDIA's ISAAC

## Disclaimer

This work is based on the NVIDIA's Omniverse Isaac. In particular, Isaac Gym (not IsaacSim).<br/>
In order to use it, you first need to download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation.
<br/><br/>
Also the core modules and functionalities are from NVIDIA's Isaac Gym [repo](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/)


## How to use
IsaacGym is only available on Ubuntu18.06, 20.04 and above at the moment. <br/>
I've implemented similar 3D bin packer logic using Stable Baselines3 for windows/Mac/or Linux, too. Please refer to the folder "bin_packer_sb3" [here](https://github.com/djkim9031/BinPacker_RL/tree/main/bin_packer_sb3) if you don't have Linux machine.<br/>
Stable Baselines3 supports vectorized environments, but the performance doens't seem to be as good.
<br/>
I highly recommend using CUDA for training/testing. Also when training, please activate wandb for real-time monitoring <br/>
<br/>
For training with PPO:
```
python3 train.py task=BinPacker train=BinPackerPPO headless=True wandb_activate=True
```

For training with SAC:
```
python3 train.py task=BinPacker train=BinPackerSAC headless=True wandb_activate=True
```

For testing (e.g., SAC):
```
python3 train.py task=BinPacker train=BinPackerSAC test=True checkpoint=runs/BinPackerSAC_xxx/nn/BinPackerSAC.pth
```

Multi GPUs training:
```
torchrun --standalone --nnodes=1 nproc_per_node=3 train.py multi_gpu=True task=BinPacker train=BinPackerPPO headless=True wandb_activate=True
```


## Work in Progress
This is still largely a work in progress. The full integration of BinPacker custom environment logic, physics sim, PPO/SAC integration to NVIDIA's Isaac Gym is complete. <br/>
However, still the optimal stacking policy is not found. I believe this is largely because the vanilla experience replay buffer for off-policy (SAC) is not suitable for this environment. And also, the reward functions are currently quite simple. More sophisticated reward function/model needs to be implemented. RLHF is also possible way forward
