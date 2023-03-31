from ppo_agent import BinPacker_ACPolicy_PPO
from sac_agent import BinPacker_ACPolicy_SAC, BinPackerFeatureExtractor
from td3_agent import BinPacker_ACPolicy_TD3
from environment import BinPacker
from vision_environment import BinPacker_Vision

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter

import gym
from gym.envs.registration import register

if __name__ == "__main__":

    boxes = {0:5, 1:4, 2:3, 3:2}
    '''
    register(
        id="BinPacker",
        entry_point="environment:BinPacker",
        kwargs={'boxes':boxes,
                'pallet_x_min':5,
                'pallet_x_max':15,
                'pallet_y_min':5,
                'pallet_y_max':15}

    )
    '''
    env = BinPacker_Vision("./xml/environment.xml",  boxes=boxes, pallet_x_min=5, pallet_x_max=15, pallet_y_min=5, pallet_y_max=15)
    
    #check_env(env)
    #env = gym.make("BinPacker")
    #env = make_vec_env("BinPacker", n_envs=4, seed=42)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="./ckpt_sac_0317/",
        name_prefix="agent",
    )   
    
    #eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path="./logs_ppo", log_path="./logs_ppo", eval_freq=50000, deterministic=True, render=False)
    
    #agent = PPO(BinPacker_ACPolicy_PPO, env, verbose=1, gamma=1.0, batch_size=256, seed=42, device="cuda", tensorboard_log="binpacker_ppo_sparse_rw")
    #agent = SAC(BinPacker_ACPolicy_SAC, env, verbose=1, gradient_steps=2, learning_starts=50000, batch_size=1024, gamma=1.0, device="cuda", seed=7, tensorboard_log="binpacker_sac_normalized_rw")
    #agent = TD3(BinPacker_ACPolicy_TD3, env, verbose=1, learning_starts=50000, batch_size=1024, gamma=1.0, device="cuda", seed=42, tensorboard_log="binpacker_td3_normalized_rw")

    #agent.learn(10000000, callback=checkpoint_callback)
    #agent.save("ckpt_sac_0314/final")
    
    policy_kwargs = dict(
        features_extractor_class = BinPackerFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=512),
        net_arch = dict(pi=[512, 1024, 512], qf=[512, 2048, 1024, 512])
    )
    
    agent = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_starts=5000, batch_size=1024, gamma=1.0, buffer_size=50000, device="cuda", seed=7, tensorboard_log="binpacker_sac_normalized_rw")
    #agent = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=1.0, batch_size=256, seed=42, device="cuda", tensorboard_log="binpacker_ppo_imagelike")
    agent.learn(10000000, callback=checkpoint_callback)
    agent.save("ckpt_sac_0317/final")
   


