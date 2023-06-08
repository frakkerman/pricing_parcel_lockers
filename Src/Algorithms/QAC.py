import numpy as np
import torch
from torch import tensor, float32, long
import torch.nn.functional as F
from Src.Utils.Utils import MemoryBuffer,Trajectory,NeuralNet
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, Actor, Critic

# This function implements the a Q-actor critic (QAC) algorithm
# contains the updates of actor and critic
class QAC(Agent):
    def __init__(self, config):
        super(QAC, self).__init__(config)

        # Obtain state features
        self.state_features = Basis.get_Basis(config=config)

        # Initialize action space matrix
        self.action_space_matrix = config.env.action_space_matrix
        
        self.mapping_fct = No_Action_representation(state_dim=self.state_features.feature_dim,
                                                                         action_dim=self.action_dim, config=config,action_space=self.action_space_matrix)

        # Initialize storage containers
        self.action_space_matrix_size = self.action_space_matrix.shape[0]

        # Initialize critic
        self.critic = Critic.Qval(state_dim=self.state_features.feature_dim,action_dim=self.config.env.n_actions, config=config)

        # Initialize actor: VAC, uses categorical, if not use Gaussian policy
        self.actor = Actor.Categorical(action_dim=self.action_space_matrix_size,state_dim=self.state_features.feature_dim, config=config,action_space=self.action_space_matrix)

        # Add a memory for training LAR and a container for training critic and actor based on single trajectories
        self.memory =   MemoryBuffer(max_len=self.config.buffer_size, state_dim=self.state_dim,
                                     action_dim=1, atype=long, config=config,
                                     dist_dim=self.mapping_fct.reduced_action_dim)  # off-policy
        self.trajectory = Trajectory(max_len=self.config.batch_size, state_dim=self.state_dim,
                                     action_dim=1, atype=long, config=config,
                                     dist_dim=self.mapping_fct.reduced_action_dim)  # on-policy

        # Define learning modules -- in LAR we have 3 learning modules as we have to additionally train the SL predictor
        self.modules = [('actor', self.actor), ('critic', self.critic)]

        # Define the action space matrix as tensor
        self.action_space_matrix = torch.tensor(self.action_space_matrix,dtype=torch.float32)
        self.action_trafo = self.action_id_to_tensor
       

    def get_action(self, state,training):
        state = tensor(state, dtype=float32, requires_grad=False)
        state = self.state_features.forward( state.view(1, -1))
        a_hat, _ = self.actor.get_action(state, training)
        action = self.mapping_fct.get_best_match(a_hat,state,self.critic)

        a_hat = a_hat.cpu().view(-1).data.numpy()

        return action, a_hat

    def update(self, s1, a1, a_hat_1, r1, s2, done):
        loss_actor = 0
        loss_critic = 0
        self.memory.add(s1, a1, a_hat_1, r1, s2, int(done != 1))
        self.trajectory.add(s1, a1, a_hat_1, r1, s2, int(done != 1))
        if self.trajectory.size >= self.config.batch_size or done:
            loss_actor,loss_critic = self.optimize()
            self.trajectory.reset()
        return loss_actor,loss_critic

    def action_to_tensor(self,a1,a2):
        return a1,a2
    def action_id_to_tensor(self,a1,a2):
        id = a1.cpu().numpy()[0][0]
        action2= self.action_space_matrix[a2].view(1, -1)
        return self.action_space_matrix[id].view(1,-1),action2

    def optimize(self):
        s1, a1, a_hat_1, r1, s2, not_absorbing = self.trajectory.get_all()

        a2, a_hat_2 = self.get_action(s2,training=True)
        s1 = self.state_features.forward(s1)
        s2 = self.state_features.forward(s2)

        action1,action2 = self.action_trafo(a1,a2)

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



class No_Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 config,
                 action_space):
        super(No_Action_representation, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.config = config
        self.norm_const = np.log(self.action_dim)
        self.reduced_action_dim = action_dim

    def get_best_match(self, action,state,critic):
        action = np.where(np.all(self.action_space==action.numpy()[0],axis=1))[0][0]
        return action