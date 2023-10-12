import numpy as np
import torch
from torch import tensor, float32, long
import torch.nn.functional as F
import scipy
from Src.Utils.Utils import Trajectory
from Src.Utils import Basis, Actor, Critic
from Src.Algorithms.Agent import Agent


# This function integrates the a Q-actor critic (QAC) algorithm with the different Continuous-to-Discrete (C2D) mapping functions, i.e., VAC, knn, DNC, MinMax, LAR, and
# contains the updates of actor and critic
class PPO(Agent):
    def __init__(self, config):
        super(PPO, self).__init__(config)

        self.get_action = self.get_action_pricing
        self.max_p = config.max_price
        self.min_p = config.min_price

        # Obtain state features
        self.state_features = Basis.get_Basis(config=config)
        
        adim = 11
        if config.instance=='Austin':
            adim = 278+1
        if config.instance=='Seattle':
            adim = 299+1

        # Initialize critic
        self.critic = Critic.Qval(state_dim=self.state_features.feature_dim,action_dim=adim, config=config)
        
        # Initialize actor we use a Gaussian policy                
        self.actor = Actor.Gaussian(action_dim=adim,state_dim=self.state_features.feature_dim, config=config)
        
        # Add a memory for training critic and actor based on single trajectories
        self.trajectory = Trajectory(max_len=1, state_dim=self.state_dim,
                                     action_dim=1, atype=long, config=config,
                                     dist_dim=adim)  # on-policy

        # Define learning modules -- in LAR we have 3 learning modules as we have to additionally train the SL predictor
        self.modules = [('actor', self.actor), ('critic', self.critic)]
        self.weights_changed = True


    def get_action_pricing(self, state,training,training2=True):
        state = tensor(state, dtype=float32, requires_grad=False)
        state = self.state_features.forward( state.view(1,-1))#.view(1, -1)
        a_hat, _ = self.actor.get_action(state, training=True)

        a_hat = a_hat.cpu().view(-1).data.numpy()
    
    
        a_hat = np.clip(a_hat,self.min_p,self.max_p)
        return np.around(a_hat,decimals=2)
    
    def update(self, s1, a1, a_hat_1, r1, s2, done):
        loss_actor = 0
        loss_critic = 0

        self.trajectory.add(s1, a1, a_hat_1, r1, s2, int(done != 1))
        if self.trajectory.size >= 1 or done:
            loss_actor,loss_critic = self.optimize()
            self.weights_changed = True
            self.trajectory.reset()
        else:
            self.weights_changed = False

        return loss_actor,loss_critic


    #### PPO specifics
    # Discounted cumulative sums of vectors for computing sum of discounted rewards and advantage estimates
    def discounted_cumulative_sums(self,x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def optimize(self):
        if self.trajectory.size > 1:
            self.calculate_trajectory_information()

            for i in range(self.config.policy_update_epochs):
                ## PPO Clipping
                ratio = torch.exp(self.actor.get_log_prob(self.state_buffer[:-1,], self.a_hat_buffer[:-1,])[0].sum(axis=1)-self.logprobas)
                clip_advantage = torch.where(self.advantages[:,0] > 0, (1 + self.config.clipping_factor) * self.advantages[:,0], (1 - self.config.clipping_factor) * self.advantages[:,0]) # Apply clipping
                loss_actor = -torch.mean(torch.minimum(ratio * self.advantages[:,0],clip_advantage))  # Setup loss function as mean of individual L_clip and negative sign, as tf minimizes by default # -torch.mean(ratio*self.advantages[:,0])

                self.actor_step(loss_actor,clip_norm=1)
            for i in range(self.config.critic_update_epochs):
                predictions = self.critic.forward(self.state_buffer, self.action_buffer)
                loss_critic = F.huber_loss(self.n_step_return_buffer,predictions[:-1])
                self.critic_step(loss_critic, clip_norm=1)

            return loss_actor.cpu().data.numpy(),loss_critic.cpu().data.numpy()
        else:
            print('Error: trajectory contains too little samples')
        
    def calculate_trajectory_information(self):
        self.state_buffer, self.action_buffer, self.a_hat_buffer, reward_buffer, next_state_buffer, done_buffer = self.trajectory.get_current_transitions()
        self.state_buffer = self.state_features.forward(self.state_buffer)
        value_buffer = self.critic.forward(self.state_buffer, self.action_buffer)

        ## δ = r(s_t,a_t)+γV(s_{t+1})-V(s_t)
        self.targets = (reward_buffer[:-1] + self.config.gamma * value_buffer[1:] * done_buffer[1:]).detach()
        self.predictions = value_buffer[:-1]

        deltas = (self.targets-self.predictions).detach().numpy()

        # A(s_t,a_t) = Q(s_t,a_t)-V(s_t) = 𝔼[r(s_t,a_t)+γV(s_{t+1})|s_t,a] - A(s_t) ~ G^λ_t(s_t,a_t)-V̂(s_t) ~ Sum_{k=t}^{T} (γλ)^{k-t} δ_k, if T big
        self.advantages = torch.tensor(self.discounted_cumulative_sums(deltas, self.config.gamma * self.config.td_lambda).copy())

        # Calculate total return (i.e., sum of discounted rewards) as target for value function update
        episode_length = len(reward_buffer)
        end_of_episode_vf = np.ones(episode_length-1)
        for i in range(episode_length-1):
            end_of_episode_vf[i] = end_of_episode_vf[i]*self.config.gamma**(episode_length-1-i)
        end_of_episode_vf = value_buffer[-1].detach().numpy() * done_buffer[-1].detach().numpy() * end_of_episode_vf
        # G^n(s_t, a_t) = sum of disc rewards + value function of final next state
        self.n_step_return_buffer = torch.tensor(self.discounted_cumulative_sums(reward_buffer.detach().numpy()[:-1], self.config.gamma).copy()[:,0]+end_of_episode_vf,dtype=torch.float32).view(-1,1)

        self.logprobas = self.actor.get_log_prob(self.state_buffer[:-1,],self.a_hat_buffer[:-1,])[0].sum(axis=1).detach()