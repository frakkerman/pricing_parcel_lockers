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
        returns = []
        total_loss_actor_history = []
        total_loss_critic_history = []
        total_loss_actor = 0
        total_loss_critic = 0

        checkpoint = self.config.save_after
        rm, start_ep = 0, 0

        t0 = time()
        self.episode = 0
        for episode in range(start_ep, self.config.max_episodes):

            episode_loss_actor = []
            episode_loss_critic = []

            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                action, a_hat = self.model.get_action(self.env.abstract_state_ppo(state), training=True)
               # new_state, reward, done, info = self.env.step(action=action)
                new_state, done, stats,route_data  = self.env.step(action=action)
                reward=0.01#reward during run = 0.001 
                loss_actor,loss_critic=self.model.update(self.env.abstract_state_ppo(state), action, a_hat, reward, self.env.abstract_state_ppo(new_state), done)
                episode_loss_actor.append(loss_actor)
                episode_loss_critic.append(loss_critic)
                state = new_state
                total_r += reward
                step += 1
                if step >= self.max_steps or done:
                    travel_time = self.model.update_route(route_data,state,True)#do full update when episode is done
                    reward = -Utils.total_costs(stats[1],stats[2],travel_time,stats[3],stats[6],self.config)
                    loss_actor,loss_critic=self.model.update(self.env.abstract_state_ppo(state), action, a_hat, reward, self.env.abstract_state_ppo(new_state), done)
                    episode_loss_actor.append(loss_actor)
                    episode_loss_critic.append(loss_critic)
                    total_r += reward
                    break

            # Track actor and critic loss with smoothing
            total_loss_actor = total_loss_actor*0.99+0.01*np.average(episode_loss_actor)
            total_loss_actor_history.append(total_loss_actor)
            total_loss_critic=total_loss_critic*0.99+0.01*np.average(episode_loss_critic)
            total_loss_critic_history.append(total_loss_critic)
            rm = 0.9*rm + 0.1*total_r

            if episode%checkpoint == 0 or episode == self.config.max_episodes-1:
                print('time required for '+str(checkpoint)+' :' +str(time()-t0))
                print('Episode '+str(episode)+' / current actor loss: ' + str(total_loss_actor))
                print('Episode '+str(episode)+' / current critic loss: ' + str(total_loss_critic))
                returns.append(total_r)
                Utils.plot_training_curves(returns,config=self.config)
                t0 = time()
    

    def eval(self, episodes=1):
         # Evaluate the model
         travel_time = []
         total_cost = []
         actions = []
         accepted_price = []
         accepted_discount = []
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
                 action, a_hat = self.model.get_action(self.test_env.abstract_state_ppo(state), training=True)
                 new_state, done, stats,route_data  = self.test_env.step(action=action)
                 actions.append([*action,step,episode])
                 # accepted_price.append([stats[3],step,episode])
                 home_delivery_loc.append([stats[5],step,episode])
                 state = new_state
                 step += 1
                 step_time.append(time()-t1)
                 if step >= self.max_steps:
                     break
             travel_time.append([self.env.reopt_for_eval(route_data),episode])#short HGS (re-opt) call
            # total_cost.append([Utils.total_costs(stats[1],stats[2],travel_time,stats[3],self.config)[0][0][0],episode])
             service_time.append([stats[2],episode])
             count_home_delivery.append([stats[1],episode])
             accepted_price.append(stats[3])
             accepted_discount.append(stats[6])
             for i in stats[4]:
                 parcel_lockers_remaining_capacity.append([i.remainingCapacity,i.location.x,i.location.y,episode])
             
         cnt=0
         trvl=0
         trvl_list=[]
         srvc=0
         fail=0
         distance=0
         count_pp=0
         for i in home_delivery_loc:
             if i[0]>0:
                 count_pp+=1
             distance += i[0]
       
         
         cost_multiplier = (self.config.driver_wage+self.config.fuel_cost*self.config.truck_speed) / 3600
         for i in range(0,len(count_home_delivery)):
             cnt += count_home_delivery[i][0]
             trvl += (travel_time[i][0]*cost_multiplier)
             trvl_list.append(travel_time[i][0]*cost_multiplier)
             srvc += (service_time[i][0]*cost_multiplier)
             fail += count_home_delivery[i][0]*self.config.home_failure*self.config.failure_cost#costs of failed delivery
         
         d_list = np.concatenate(accepted_discount)
         r_list = np.concatenate(accepted_price)
         print('percentage home delivery: ', cnt/len(home_delivery_loc))
         print('travel costs: ', trvl/episodes)
         print('service costs: ', srvc/episodes)
         print('failure costs: ', fail/episodes)
         print('Avg. Charge: ', np.mean(r_list), 'std.: ', np.std(r_list))
         print('Avg. Discount: ', -np.mean(d_list), 'std.: ', np.std(d_list))
         print('Charge revenue: ', np.sum(r_list)/episodes)
         print('Discount costs: ', -np.sum(d_list)/episodes)
         print('total costs: ', (trvl+srvc+fail-np.sum(d_list)-np.sum(r_list))/episodes)
         print('average travelled by customers: ', distance/count_pp)
         
         # print(trvl_list)
         
            
         
         
            
         
        #directly save statistics
         # Utils.save_eval_stats(travel_time,total_cost,actions,accepted_price,count_home_delivery,service_time,
         #                       parcel_lockers_remaining_capacity,home_delivery_loc,step_time,self.config)
             
         return total_cost, accepted_price,step_time



def main(train=True):
    t = time()
    args = Parser().get_parser().parse_args()

    config = Config(args)
    solver = Solver(config=config)

    if train:
        solver.train()
    
    #evaluate model
    rewards,prices,step_time = solver.eval(30)  
  #  Utils.plot_test_boxplot(rewards,prices,step_time,config)
    
    print('total timing: ', time()-t)

if __name__== "__main__":
        main(train=True)

