import numpy as np
import torch
from torch import tensor, float32, long
import torch.nn.functional as F
import scipy
from Src.Utils.Utils import Trajectory,get_dist_mat_HGS,extract_route_HGS
from Src.Utils import Basis, Actor, Critic
from Src.Algorithms.Agent import Agent
from hygese import AlgorithmParameters, Solver


# This function integrates Gaussian PPO, as proposed in Schulman et al. (2017)
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
            
        self.adjacency = config.adjacency
        self.dist_matrix = config.dist_matrix
        self.load_data = config.load_data
        #hgs settings
        ap_final = AlgorithmParameters(timeLimit=config.hgs_final_time)  # seconds
        self.hgs_solver_final = Solver(parameters=ap_final, verbose=False)#used for final route     
            
        self.state_dim = self.state_features.observation_space.shape[0]
        

        # Initialize critic
        self.critic = Critic.Qval(state_dim=self.state_features.feature_dim,action_dim=adim, config=config)
        
        # Initialize actor we use a Gaussian policy                
        self.actor = Actor.Gaussian(action_dim=adim,state_dim=self.state_features.feature_dim, config=config)
        
        # Add a memory for training critic and actor based on single trajectories
        self.trajectory = Trajectory(max_len=self.config.batch_size_ppo, state_dim=self.state_dim,
                                     action_dim=adim, atype=long, config=config,
                                     dist_dim=adim)  # on-policy

        # Define learning modules -- in LAR we have 3 learning modules as we have to additionally train the SL predictor
        self.modules = [('actor', self.actor), ('critic', self.critic)]
        self.weights_changed = True


    def get_action_pricing(self, state,state2,training):
        state = tensor(state, dtype=float32, requires_grad=False)
        state = self.state_features.forward( state.view(1,-1))#.view(1, -1)
        a_hat, _ = self.actor.get_action(state, training=True)

        a_hat = a_hat.cpu().view(-1).data.numpy()
    
    
        a_hat_clip = np.clip(a_hat,self.min_p,self.max_p)

        return np.around(a_hat_clip,decimals=2),a_hat_clip
    
    def update(self, s1, a1, a_hat_1, r1, s2, done):
        loss_actor = 0
        loss_critic = 0
        
        self.trajectory.add(s1, a1, a_hat_1, r1, s2, int(done != 1))
        if  self.trajectory.size >= self.config.batch_size_ppo or done:
            loss_actor,loss_critic,self.weights_changed = self.optimize()
            if self.weights_changed:
                self.trajectory.reset()
        else:
            self.weights_changed = False

        return loss_actor,loss_critic
    
    
    def update_route(self,data,state,done):
        if not done:
            return 0.0
        else:
            #obtain final CVRP schedule after end of booking horizon
            if self.load_data:
                data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix,data['id'])
            fleet,cost = self.reopt_HGS_final(data)#do a final reopt
            return cost


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
                loss_actor = torch.mean(torch.minimum(ratio * self.advantages[:,0],clip_advantage))  # Setup loss function as mean of individual L_clip and negative sign, as tf minimizes by default # -torch.mean(ratio*self.advantages[:,0])

                self.actor_step(loss_actor,clip_norm=1)
            for i in range(self.config.critic_update_epochs):
                predictions = self.critic.forward(self.state_buffer, self.action_buffer)
                loss_critic = F.huber_loss(self.n_step_return_buffer,predictions[:-1])
                self.critic_step(loss_critic, clip_norm=1)

            return loss_actor.cpu().data.numpy(),loss_critic.cpu().data.numpy(),True
        else:
            # loss_actor,loss_critic = self.QAC_optimize()
            return 0,0,False
        
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
        
        
    def QAC_optimize(self):
        s1, a1, a_hat_1, r1, s2, not_absorbing = self.trajectory.get_current_transitions()
        
        a2, a_hat_2 = self.get_action(s2,training=True)
        s1 = self.state_features.forward(s1)
        s2 = self.state_features.forward(s2)

        action1,action2 = self.action_to_tensor(a1,a2)

        # Define critic loss
        next_val = self.critic.forward(s2,action2).detach()
        val_exp  = r1 + self.config.gamma * next_val * not_absorbing
        val_pred = self.critic.forward(s1,action1)
        loss_critic = F.huber_loss(val_pred, val_exp) # F.mse_loss

        # Define actor loss
        # see https://github.com/pytorch/pytorch/issues/9442 for log of pdf
        td_error = (val_exp-val_pred).detach()
        logp, dist = self.actor.get_log_prob(s1, a_hat_1)
        loss_actor = -1.0 * torch.sum(td_error*logp)

        # Take one policy gradient step
        loss = loss_actor + loss_critic
        self.step(loss, clip_norm=1)

        return loss_actor.cpu().data.numpy(),loss_critic.cpu().data.numpy()

    def action_to_tensor(self,a1,a2):
        return tensor(a1, dtype=float32, requires_grad=False),tensor(a2, dtype=float32, requires_grad=False)
        
        
    def reopt_HGS_final(self,data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0#depot demand=0
        result = self.hgs_solver_final.solve_cvrp(data)  
        #update current routes
        fleet = extract_route_HGS(result,data)
        return fleet,result.cost