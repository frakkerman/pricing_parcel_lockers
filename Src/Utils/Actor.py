import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from Src.Utils.Utils import NeuralNet

# This file implements all neural network functions required to construct the actor
class Actor(NeuralNet):
    def __init__(self, state_dim, config, action_dim):
        super(Actor, self).__init__()

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

    def init(self,config):
        temp, param_list = [], []
        for name, param in self.named_parameters():
            temp.append((name, param.shape))
            if 'var' in name:
                param_list.append(
                    {'params': param, 'lr': config.actor_lr / 100})
            else:
                param_list.append({'params': param})
        self.optim = config.optim(param_list, lr=config.actor_lr)

        print("Actor: ", temp)

#this implementation is based on the implementation from the paper:
#Dynamic Neighborhood Construction for Structured Large Discrete Action Spaces, by Akkerman et al. (2023)
class Gaussian(Actor):
    def __init__(self, state_dim, action_dim, config):
        super(Gaussian, self).__init__(action_dim,state_dim, config)

        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, config.hiddenActorLayerSize)
        self.fc2 = nn.Linear(config.hiddenActorLayerSize, config.hiddenActorLayerSize)
        self.fc3 = nn.Linear(config.hiddenActorLayerSize, self.action_dim)
        self.relu = nn.ReLU()

        self.output_layer = torch.sigmoid
        
        self.gauss_variance = config.gauss_variance
        
        self.action_multiplier = np.full(action_dim, config.min_price )
        self.action_multiplier[0] = config.max_price
        self.action_multiplier = torch.tensor(self.action_multiplier,requires_grad=False)
       
        self.init(config)

    def forward(self, state):
        mean = self.fc1(state)
        mean = self.relu(mean)
        mean = self.fc2(mean)
        mean = self.relu(mean)
        mean = self.fc3(mean)

        mean = self.output_layer(mean) * self.action_multiplier 

        var = torch.ones_like(mean, requires_grad=False) * self.gauss_variance
        return mean, var


    def get_action(self, state, training):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        if training:
            action = dist.sample()
        else:
            action = mean

        return action, dist

    def get_log_prob(self, state, action):
        mean, var = self.forward(state)
        dist = Normal(mean, var)
        return dist.log_prob(action), dist

    def get_log_prob_from_dist(self, dist, action):
        return dist.log_prob(action)

    def get_prob_from_dist(self, dist, action, scalar=True):
        if scalar:
            prod = torch.exp(torch.sum(dist.log_prob(action), -1, keepdim=True))
        else:
            prod = torch.exp(dist.log_prob(action))
        return prod

