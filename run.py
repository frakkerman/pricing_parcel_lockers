import numpy as np
import Src.Utils.Utils as Utils
from Src.parser import Parser
from Src.config import Config
from time import time

"""
TODO: just use cvrp standrad instance files
TODO: new historic routes
TODO: CNN + state embedding/representation using grid overlay and time dimension
TODO: CNN training loop + graph
make run plan, run with different instanbce size+parcelpoint density
perhaps add appartment builidng delivery time?
question, how to determine costs per customer, how does the data structure look?
also study fixed fee/free home delivery
add time windows to support complex architecture?...
"""

class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]
        
        #to ensure we do not exceed the fleet capacity
        self.max_steps = int(config.n_vehicles*config.veh_capacity)-1
        
        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.model = config.algo(config=config)

    def eval(self, episodes=1):
        # Evaluate the model
        costs = []
        steps = []
        step_time = []
        for episode in range(episodes):
            state = self.env.reset(training=False)#np.float32(self.env.reset(training=False))
            step = 0
            done = False
            while not done:
                t1 = time()
                action = self.model.get_action(state, training=False)
                new_state, done, info,data = self.env.step(action=action,training=False)
                state = new_state
                step += 1
                step_time.append(time()-t1)
                if step >= self.max_steps:
                    break
            costs.append(self.model.update(data))#fix
            steps.append(step)
        return costs,step_time,steps

    # Main training/simulation loop
    def train(self):
        # Learn the model on the environment
        test_returns = []
        test_std = []
        run_time = []
        run_info = []
        actions = []
        costs = []

        avg_step_time = []
        checkpoint = self.config.save_after
        start_ep = 0

        steps = 0
        t0 = time()
        t_init = time()
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            step = 0
            done = False

            while not done:
                action = self.model.get_action(state, training=True)
                new_state, done, info,data = self.env.step(action=action)
                run_info.append(info)
                actions.append(action)
                state = new_state
                step += 1
                costs.append(self.model.update(data,state,done))#update model only after episode ends
                if step >= self.max_steps:
                    costs.append(self.model.update(data,state,True))#update model only after episode ends
                    break
            steps += step

            if episode%checkpoint == 0 or episode == self.config.max_episodes-1:
                print('time required for '+str(checkpoint)+' episodes :' +str(time()-t0))
                # test_reward, step_time, _ = self.eval(10)   
                # avg_test_reward = np.average(test_reward)
                # std_test_reward = np.std(test_reward)
                # avg_step_time.append(np.average(step_time))
                # test_std.append(std_test_reward)
                # test_returns.append(avg_test_reward)
                # Utils.save_plots_test_runs(test_returns,test_std,avg_step_time,config=self.config)
                run_time.append((time()-t_init))
                Utils.save_plots_stats(run_info,costs,run_time,actions=actions,config=self.config,episode=episode)
               
                t0 = time()
                steps = 0

def main(train=True):
    t = time()
    args = Parser().get_parser().parse_args()

    config = Config(args)
    solver = Solver(config=config)

    if train:
        solver.train()
    print(time()-t)

if __name__== "__main__":
        main(train=True)

