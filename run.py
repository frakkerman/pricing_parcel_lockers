import numpy as np
import Src.Utils.Utils as Utils
from Src.parser import Parser
from Src.config import Config
from time import time

"""
TODO: perhaps add VRPTW later (would require use of pyvrp lib), would make 3d conv interesting
check all methods (clustering going okay?)
check settings of MNL model
implement Lin reg, MLP, and ablation option
Implement PPO
robustness option experiment?
"""

class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env#env used for training
        self.test_env = self.config.test_env#seperate env used for testing only
        self.state_dim = np.shape(self.env.reset())[0]
        
        #to ensure we do not exceed the fleet capacity
        self.max_steps = int(config.n_vehicles*config.veh_capacity)-1
        
        print("State space: {}".format(self.state_dim))

        self.model = config.algo(config=config)

    # Main training/simulation loop
    def train(self):
        # Learn the model on the environment
        rewards = []

        checkpoint = self.config.save_after
        start_ep = 0

        steps = 0
        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            step = 0
            done = False

            while not done:
                action = self.model.get_action(state, training=True)
                new_state, done, stats,route_data = self.env.step(action=action)
                state = new_state
                step += 1
                _ = self.model.update(route_data,state,done)
                if step >= self.max_steps:
                    travel_time = self.model.update(route_data,state,True)#do full update when episode is done
                    rewards.append(Utils.total_costs(stats[1],stats[2],travel_time,stats[3],self.config))
                    break
            steps += step
            
            if episode%checkpoint == 0 or episode == self.config.max_episodes-1:
                print('time required for '+str(checkpoint)+' episodes :' +str(time()-t0))
                Utils.plot_training_curves(rewards,self.config)
                #Utils.save_plots_stats(run_stats,travel_time,run_time,actions=actions,config=self.config,episode=episode)
               
                t0 = time()
                steps = 0
            
        #evaluate model
        # rewards,step_time = eval()  
        # Utils.plot_test_boxplot(rewards,step_time,self.config)

    def eval(self, episodes=1):
         # Evaluate the model
         travel_time = []
         total_cost = []
         actions = []
         accepted_price = []
         count_home_delivery = []
         service_time = []
         parcel_lockers_remaining_capacity = []
         home_delivery_loc = []
         step_time = []
         for episode in range(episodes):
             state = self.test_env.reset()
             step = 0
             done = False
             while not done:
                 t1 = time()
                 action = self.model.get_action(state, training=False)
                 new_state, done, stats,route_data = self.test_env.step(action=action)
                 actions.append([*action,step,episode])
                 accepted_price.append([stats[3],step,episode])
                 home_delivery_loc.append([stats[5],stats[6],stats[7],stats[8],step,episode])
                 state = new_state
                 step += 1
                 step_time.append(time()-t1)
                 if step >= self.max_steps:
                     break
             travel_time.append([self.env.reopt_for_eval(route_data),episode])#short HGS (re-opt) call
             total_cost.append([Utils.total_costs(stats[1],stats[2],travel_time,stats[3],self.config)[0][0][0],episode])
             service_time.append([stats[2],episode])
             count_home_delivery.append([stats[1],episode])
             for i in stats[4]:
                 parcel_lockers_remaining_capacity.append([i.remainingCapacity,i.location.x,i.location.y,episode])
             
         #directly save statistics
         Utils.save_eval_stats(travel_time,total_cost,actions,accepted_price,count_home_delivery,service_time,
                               parcel_lockers_remaining_capacity,home_delivery_loc,step_time,self.config)
             
         return total_cost, accepted_price,step_time



def main(train=True):
    t = time()
    args = Parser().get_parser().parse_args()

    config = Config(args)
    solver = Solver(config=config)

    if train:
        solver.train()
    
    #evaluate model
    rewards,prices,step_time = solver.eval(2)  
    Utils.plot_test_boxplot(rewards,prices,step_time,config)
    
    print(time()-t)

if __name__== "__main__":
        main(train=True)

