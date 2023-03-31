from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.network_builder import NetworkBuilder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.algos_torch.sac_helper import  SquashedNormal
from rl_games.common.layers.recurrent import  GRUWithDones, LSTMWithDones


class BinPacker_SAC_Actor(NetworkBuilder.BaseNetwork):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, backbone, output_dim, log_std_bounds, bMultiDim, **mlp_args):
        super().__init__()

        self.log_std_bounds = log_std_bounds

        self.backbone = backbone
        self.multidim = bMultiDim
        self.isD2RL = mlp_args['d2rl']

        self.trunk = self._build_mlp(**mlp_args)
        if(self.isD2RL):
            last_layer = list(self.trunk.children())[-2][-1].out_features
            self.last_layer = nn.Linear(last_layer, output_dim)
        else:
            last_layer = list(self.trunk.children())[-2].out_features
            self.trunk = nn.Sequential(*list(self.trunk.children()), nn.Linear(last_layer, output_dim))

    def forward(self, obs):
        if(self.multidim):
            state = obs.clone()
        else:
            state = obs.unsqueeze(1).clone()

        feature = self.backbone(state)
        feature = feature.contiguous().view(feature.size(0), -1)
        feature = torch.cat([feature, obs], dim=-1)

        if(self.isD2RL):
            out = self.trunk.forward(feature)
            mu, log_std = self.last_layer(out).chunk(2, dim=-1)
        else:
            mu, log_std = self.trunk(feature).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        #log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, log_std_min, log_std_max)
        #log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        # TODO: Refactor

        dist = SquashedNormal(mu, std)
        # Modify to only return mu and std
        return dist


class BinPacker_SAC_DoubleQCritic(NetworkBuilder.BaseNetwork):
    """Critic network, employes double Q-learning."""
    def __init__(self, backbone, output_dim, bMultiDim, **mlp_args):
        super().__init__()

        self.backbone = backbone
        self.multidim = bMultiDim
        self.isD2RL = mlp_args['d2rl']

        self.Q1 = self._build_mlp(**mlp_args)
        if(self.isD2RL):
            last_layer = list(self.Q1.children())[-2][-1].out_features
            self.Q1_last_layer = nn.Linear(last_layer, output_dim)
        else:
            last_layer = list(self.Q1.children())[-2].out_features
            self.Q1 = nn.Sequential(*list(self.Q1.children()), nn.Linear(last_layer, output_dim))

        self.Q2 = self._build_mlp(**mlp_args)
        if(self.isD2RL):
            last_layer = list(self.Q2.children())[-2][-1].out_features
            self.Q2_last_layer = nn.Linear(last_layer, output_dim)
        else:
            last_layer = list(self.Q2.children())[-2].out_features
            self.Q2 = nn.Sequential(*list(self.Q2.children()), nn.Linear(last_layer, output_dim))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        if(self.multidim):
            state = obs.clone()
        else:
            state = obs.unsqueeze(1).clone()

        feature = self.backbone(state)
        feature = feature.contiguous().view(feature.size(0), -1)
        obs_action = torch.cat([feature, obs, action], dim=-1)

        if(self.isD2RL):
            out1 = self.Q1.forward(obs_action)
            out2 = self.Q2.forward(obs_action)

            q1 = self.Q1_last_layer(out1)
            q2 = self.Q2_last_layer(out2)
        else:
            q1 = self.Q1(obs_action)
            q2 = self.Q2(obs_action)

        return q1, q2
    
class BinPacker_SACBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = BinPacker_SACBuilder.Network(self.params, **kwargs)
        return net

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape_orig = kwargs.pop('input_shape')

            obs_dim = kwargs.pop('obs_dim')
            action_dim = kwargs.pop('action_dim')
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            if(not self.multidim):
                input_shape = (1, input_shape_orig[0])
            else:
                input_shape = input_shape_orig

            cnn_args = {
                    'ctype' : self.backbone_conv['type'], 
                    'input_shape' : input_shape, 
                    'convs' :self.backbone_conv['convs'], 
                    'activation' : self.backbone_conv['activation'], 
                    'norm_func_name' : self.backbone_conv['normalization'],
                }
            
            print("Building Actor")
            backbone_actor = self._build_conv(**cnn_args)

            with torch.no_grad():
                if(not self.multidim):
                    n_flatten_shape = backbone_actor(torch.zeros(1, 1, input_shape_orig[0])).shape
                    n_flatten = n_flatten_shape[1]*n_flatten_shape[2]
                else:
                    n_flatten_shape = backbone_actor(torch.zeros(1, input_shape_orig[0], input_shape_orig[1], input_shape_orig[2])).shape
                    n_flatten = n_flatten_shape[1]*n_flatten_shape[2]*n_flatten_shape[3]

            actor_mlp_args = {
                'input_size' : n_flatten + input_shape_orig[0], 
                'units' : self.mlp_units, 
                'activation' : self.mlp_activation, 
                'norm_func_name' : self.mlp_normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }

            critic_mlp_args = {
                'input_size' : n_flatten + input_shape_orig[0] + action_dim, 
                'units' : self.mlp_units, 
                'activation' : self.mlp_activation, 
                'norm_func_name' : self.mlp_normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            
            self.actor = self._build_actor(backbone_actor, 2*action_dim, self.log_std_bounds, **actor_mlp_args)

            if self.separate:
                print("Building Critic")
                backbone_critic = self._build_conv(**cnn_args)
                self.critic = self._build_critic(backbone_critic, 1, **critic_mlp_args)
                print("Building Critic Target")
                backbone_critic_target = self._build_conv(**cnn_args)
                self.critic_target = self._build_critic(backbone_critic_target, 1, **critic_mlp_args)
                self.critic_target.load_state_dict(self.critic.state_dict())  

            cnn_init = self.init_factory.create(**self.backbone_conv['initializer'])
            mlp_init = self.init_factory.create(**self.mlp_initializer)
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

        def _build_critic(self, backbone, output_dim, **mlp_args,):
            return BinPacker_SAC_DoubleQCritic(backbone, output_dim, self.multidim, **mlp_args)

        def _build_actor(self, backbone, output_dim, log_std_bounds, **mlp_args):
            return BinPacker_SAC_Actor(backbone, output_dim, log_std_bounds, self.multidim, **mlp_args)
        
        def forward(self, obs_dict):
            """TODO"""
            obs = obs_dict['obs']
            mu, sigma = self.actor(obs)
            return mu, sigma
 
        def is_separate_critic(self):
            return self.separate
        
        def load(self, params):
            self.separate = params.get('separate', True)
            self.multidim = params['multidim']

            self.backbone_conv = params['backbone']

            self.mlp_units = params['mlp']['units']
            self.mlp_activation = params['mlp']['activation']
            self.mlp_initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.mlp_normalization = params['mlp']['normalization']
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.log_std_bounds = params.get('log_std_bounds', None)

            if self.has_space:
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False

