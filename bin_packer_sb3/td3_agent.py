from typing import Any, Dict, List, Optional, Type, Union, Tuple

import torch as th
from gym import spaces
import gym
from torch import nn

from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule


class Actor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)

        # Backbone
        self.backbone = nn.Sequential(

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
        nn.Linear(540, features_dim),
        nn.ReLU()
        )
        
        self.mlp = nn.Sequential(

            nn.Linear(features_dim + observation_space.shape[0], 2*features_dim),
            nn.ReLU(),
            nn.Linear(2*features_dim, features_dim), 
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

        # Deterministic action
        self.mu = nn.Sequential(
            nn.Linear(features_dim, action_dim),
            nn.Tanh()
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features1 = obs.unsqueeze(1).clone()
        features2 = obs.clone()
        
        extractedFeat = self.backbone(features1)
        policy_input = th.cat([extractedFeat, features2], dim=1)
        latent_pi = self.mlp(policy_input)
        
        return self.mu(latent_pi)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)


class DenseContinuousCritic(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        
        # Backbone
        self.backbone = nn.Sequential(

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
        nn.Linear(540, features_dim),
        nn.ReLU()
        )

        self.share_features_extractor = False
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = nn.Sequential(

            nn.Linear(features_dim + observation_space.shape[0] + 2, 4*features_dim),
            nn.ReLU(),
            nn.Linear(4*features_dim, 2*features_dim), 
            nn.ReLU(),
            nn.Linear(2*features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1)
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, state: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        
        features1 = state.unsqueeze(1).clone()
        features2 = state.clone()

        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            extractedFeat = self.extract_features(features1)
        
        extractedFeat = self.backbone(features1)
            
        qvalue_input = th.cat([extractedFeat, features2, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, state: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        features1 = state.unsqueeze(1).clone()
        features2 = state.clone()
        
        with th.no_grad():
            extractedFeat = self.backbone(features1)
        return self.q_networks[0](th.cat([extractedFeat, features2, actions], dim=1))


class BinPacker_ACPolicy_TD3(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(self, *args, **kwargs):
        super(BinPacker_ACPolicy_TD3, self).__init__(
            *args,
            **kwargs,
        )
        
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DenseContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DenseContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)


    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode