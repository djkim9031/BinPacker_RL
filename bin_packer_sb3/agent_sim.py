from ppo_agent import BinPacker_ACPolicy_PPO
from sac_agent import BinPacker_ACPolicy_SAC, BinPackerFeatureExtractor
from environment import BinPacker
from vision_environment import BinPacker_Vision

from stable_baselines3 import PPO, SAC

if __name__ == "__main__":
    
    boxes = {0:5, 1:4, 2:3, 3:2}
    env = BinPacker_Vision("./xml/environment.xml",  boxes=boxes, pallet_x_min=5, pallet_x_max=15, pallet_y_min=5, pallet_y_max=15)

    #model = PPO(BinPacker_ACPolicy_PPO, env, verbose=1, gamma=1.0, batch_size=256, seed=42, device="cuda", tensorboard_log="binpacker_ppo")
    #model = SAC(BinPacker_ACPolicy_SAC, env, verbose=1, learning_starts=50000, batch_size=1024, gamma=1.0, device="cuda", seed=42, tensorboard_log="binpacker_sac")
    #model = SAC(BinPacker_ACPolicy_SAC, env, verbose=1, learning_starts=50000, batch_size=1024, gamma=1.0, device="cuda", seed=7, tensorboard_log="binpacker_sac_normalized_rw")
    policy_kwargs = dict(
        features_extractor_class = BinPackerFeatureExtractor,
        features_extractor_kwargs = dict(features_dim=512),
        net_arch = dict(pi=[512, 1024, 512], qf=[512, 2048, 1024, 512])
    )
    model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_starts=5000, batch_size=1024, gamma=1.0, buffer_size=10, device="cuda", seed=7, tensorboard_log="binpacker_sac_normalized_rw")
    
    agent = model.load("ckpt_sac_0317/agent_50000_steps.zip")
    
    env.render(agent=agent)
    