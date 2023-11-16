import numpy as np
import Src.Utils.Utils as Utils
from Src.parser import Parser
from Src.config import Config
from time import time

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
                _ = self.model.update(route_data,state,False)
                if step >= self.max_steps or done:
                    travel_time = self.model.update(route_data,state,True)#do full update when episode is done
                    rewards.append(Utils.total_costs(stats[1],stats[2],travel_time,stats[3],stats[6],self.config))
                    break
            
            if episode%checkpoint == 0 or episode == self.config.max_episodes-1:
                print('time required for '+str(checkpoint)+' episodes :' +str(time()-t0))
                Utils.plot_training_curves(rewards,self.config)
                #Utils.save_plots_stats(run_stats,travel_time,run_time,actions=actions,config=self.config,episode=episode)
               
                t0 = time()
    

    def eval(self, episodes=1):
         
        # Evaluate the model, see run_ppo - eval() for some interesting statistics to save,
        # we removed these statistics tracking from run.py to make the code a bit more readable,
        # but you can add those statistics tracking easily to this file again
         total_cost = []
         actions = []
         accepted_price = []
         price_time=[]
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
                 price_time.append([stats[7],step])
                 actions.append([*action,step,episode])
                 home_delivery_loc.append([stats[5],step,episode])
                 state = new_state
                 step += 1
                 step_time.append(time()-t1)
                 if step >= self.max_steps:
                     break
         
         #directly save statistics
         #Utils.save_eval_stats(travel_time,total_cost,actions,accepted_price,count_home_delivery,service_time,
                  #              parcel_lockers_remaining_capacity,home_delivery_loc,step_time,price_time,self.config)
             
         return total_cost, accepted_price,step_time



def main(train=True):
    t = time()
    args = Parser().get_parser().parse_args()

    config = Config(args)
    solver = Solver(config=config)

    if train:
        solver.train()
    
    #evaluate model
    #rewards,prices,step_time = solver.eval(20)  
    #Utils.plot_test_boxplot(rewards,prices,step_time,config)
    
    print('total timing: ', time()-t)

if __name__== "__main__":
        main(train=True)

