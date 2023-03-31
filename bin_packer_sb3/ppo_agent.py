from gym import spaces
import torch as th
from torch import nn

from typing import Callable, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy

class BinPacker_Network_PPO(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        observation_space_size: int,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Backbone
        self.backbone_policy = nn.Sequential(

        nn.Conv1d(1, 5, kernel_size=6, stride=6, bias=True),
        nn.BatchNorm1d(5),
        nn.ReLU(),
        nn.Conv1d(5, 10, kernel_size=6, stride=1, bias=True),
        nn.BatchNorm1d(10),
        nn.ReLU(),
        nn.Conv1d(10, 30, kernel_size=6, stride=1, bias=True),
        nn.BatchNorm1d(30),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(540, feature_dim),
        nn.ReLU()
        )
        
        self.backbone_value = nn.Sequential(

        nn.Conv1d(1, 5, kernel_size=6, stride=6, bias=True),
        nn.BatchNorm1d(5),
        nn.ReLU(),
        nn.Conv1d(5, 10, kernel_size=6, stride=1, bias=True),
        nn.BatchNorm1d(10),
        nn.ReLU(),
        nn.Conv1d(10, 30, kernel_size=6, stride=1, bias=True),
        nn.BatchNorm1d(30),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(540, feature_dim),
        nn.ReLU()
        )
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim + observation_space_size, 2*feature_dim),
            nn.ReLU(),
            nn.Linear(2*feature_dim, feature_dim), 
            nn.ReLU(),
            nn.Linear(feature_dim, last_layer_dim_pi),
            nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim + observation_space_size, 4*feature_dim),
            nn.ReLU(),
            nn.Linear(4*feature_dim, 2*feature_dim), 
            nn.ReLU(),
            nn.Linear(2*feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, last_layer_dim_vf),
            nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        features1 = features.unsqueeze(1).clone()
        features2 = features.clone()
        
        extractedFeat = self.backbone_policy(features1)
        policy_input = th.cat([extractedFeat, features2], dim=1)
        
        return self.policy_net(policy_input)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features1 = features.unsqueeze(1).clone()
        features2 = features.clone()
        
        extractedFeat = self.backbone_value(features1)
        value_input = th.cat([extractedFeat, features2], dim=1)
        
        return self.value_net(value_input)


class BinPacker_ACPolicy_PPO(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = BinPacker_Network_PPO(self.observation_space.shape[0], 256)

