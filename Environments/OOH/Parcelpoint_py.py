from __future__ import print_function

import numpy as np
import numpy.ma as ma
from Src.Utils.Utils import Space,get_dist_mat_HGS,get_fleet
from Environments.OOH.containers import Location,ParcelPoint,ParcelPoints,Vehicle,Fleet,Customer
from Environments.OOH.env_utils import utils_env
from Environments.OOH.customerchoice import customerchoicemodel

class Parcelpoint_py(object):
    def __init__(self,
                 model,
                 max_steps_r,
                 max_steps_p,
                 pricing = False,
                 n_vehicles=2,
                 veh_capacity=100,
                 parcelpoint_capacity=25,
                 incentive_sens=0.99,
                 base_util=0.2,
                 home_util=0.3,
                 reopt=2000,
                 load_data=False,
                 coords=[],
                 dist_matrix=[],
                 n_parcelpoints=6,
                 adjacency=[],
                 hgs_time=3.0):
        
        #episode length params
        self.max_steps = 0
        self.max_steps_r = max_steps_r
        self.max_steps_p = max_steps_p
        
        #init fleet and parcelpoints
        self.n_vehicles = n_vehicles
        self.veh_capacity = veh_capacity
        self.pp_capacity = parcelpoint_capacity
        self.data = dict()
        
        #possible passed on data
        self.coords = coords
        self.dist_matrix = dist_matrix
        self.n_parcelpoints = n_parcelpoints
        self.adjacency = adjacency
        
        #load data or generate data
        self.load_data = load_data
        self.n_unique_customer_locs = len(self.coords)-self.n_parcelpoints
        if self.load_data:
            print("\n Note: the HGS python implementation (hygese 0.0.0.8) throws an assertion error for coords<0, you will need to outcomment this check in hygese.py \n")
            self.utils = utils_env(Location,Vehicle,Fleet,ParcelPoint,ParcelPoints,self.veh_capacity,self.n_vehicles,self.pp_capacity,self.data,self.dist_matrix,hgs_time)
            self.depot = self.coords[0]
            self.parcelPoints = self.utils.get_parcelpoints_from_data(self.coords[-self.n_parcelpoints:],self.n_unique_customer_locs)
            self.action_space = Space(size=2**20)#we only select from the 20 closest parcelpoints
            self.get_customer = self.get_new_customer_from_data
            self.num_cust_loc = len(self.dist_matrix)-len(self.parcelPoints["parcelpoints"])-1
            self.dist_scaler = np.amax(self.dist_matrix)
        else:
            self.depot = Location(50,50,0,0)
            self.utils = utils_env(Location,Vehicle,Fleet,ParcelPoint,ParcelPoints,self.veh_capacity,self.n_vehicles,self.pp_capacity,self.data,self.dist_matrix,hgs_time)
            self.parcelPoints = self.utils.get_parcelpoints()
            self.action_space = Space(size=2**self.n_parcelpoints)
            self.get_customer = self.generate_new_customer
            self.dist_scaler = 10
                
        #customers
        self.home_util = home_util
        self.incentive_sens = incentive_sens
       
        self.newCustomer = Customer
        self.fleet = get_fleet([self.depot,self.depot],self.n_vehicles,self.veh_capacity)

        #pricing of offering problem variant
        if pricing:
            #self.action_space_matrix = self.get_actions(pricing,self.n_parcelpoints)
            self.customerchoice = customerchoicemodel(base_util,self.dist_scaler,self.utils.getdistance_euclidean,self.dist_matrix,self.n_unique_customer_locs)
            self.customerChoice = self.customerchoice.customerchoice_pricing
            self.get_delivery_loc = self.get_delivery_loc_pricing
        else:
            #self.action_space_matrix = self.get_actions(pricing,self.n_parcelpoints)
            self.customerchoice = customerchoicemodel(base_util,self.dist_scaler,self.utils.getdistance_euclidean,self.dist_matrix,self.n_unique_customer_locs)
            self.customerChoice = self.customerchoice.customerchoice_offer
            self.get_delivery_loc = self.get_delivery_loc_offer
        
        self.steps = 0
        self.max_steps = (self.n_vehicles*self.veh_capacity)-1
        self.reopt_freq = reopt
        
        self.reset()

    def seed(self, seed):
        self.seed = seed

    def reset(self,training=True):
        """
        Sets the environment to default conditions
        """
        self.max_steps = np.random.negative_binomial(self.max_steps_r,self.max_steps_p)
        
        self.fleet = self.utils.reset_fleet(self.fleet,[self.depot,self.depot])
        self.parcelPoints = self.utils.reset_parcelpoints(self.parcelPoints)
            
        self.steps = 0
        
        self.data['x_coordinates'] = self.depot.x
        self.data['y_coordinates'] =  self.depot.y
        self.data['id'] = 0
        self.data['time'] = 0
        self.data['vehicle_capacity'] = self.veh_capacity
        self.data['num_vehicles'] = self.n_vehicles
        
        self.curr_state = self.make_state()
        return self.curr_state

    def get_new_customer_from_data(self):
        idx = np.random.randint(1, self.num_cust_loc)
        home = self.coords[idx]#depot = 0
        home.time=self.steps
        return Customer(home,self.incentive_sens,self.home_util,idx)

    def generate_new_customer(self):
        home = Location(np.random.randint(0,500),np.random.randint(0,500),0,self.steps)
        return Customer(home,self.incentive_sens,self.home_util,0)

    def make_state(self):
        self.newCustomer = self.get_customer()
        state = [self.newCustomer,self.fleet,self.parcelPoints,self.steps]
        return state
        
    def is_terminal(self):
        if self.steps > self.max_steps:
            return 1
        else:
            return 0
    
    def get_delivery_loc_pricing(self,action):
        mask = ma.masked_array(self.parcelPoints["parcelpoints"], mask=self.adjacency[self.newCustomer.id_num])#only offer 20 closest
        return self.customerChoice(self.newCustomer,action,mask)
    
    def get_delivery_loc_offer(self,action):
        #get the chosen delivery location
        return self.customerChoice(self.newCustomer,action,self.parcelPoints["parcelpoints"])
    
    def reopt_for_eval(self,data):
        if self.load_data:
            data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix,data['id'])
        _,cost = self.utils.reopt_HGS(data)
        return cost
    
    def step(self,action,training=True):
        done=False
        self.steps += 1
        
        #get the customer's choice of delivery location
        loc,accepted,idx,price = self.get_delivery_loc(action)
        self.data['x_coordinates']= np.append(self.data['x_coordinates'],loc.x)
        self.data['y_coordinates'] = np.append(self.data['y_coordinates'],loc.y)
        self.data['id'] = np.append(self.data['id'],loc.id_num)
        self.data['time'] = np.append(self.data['time'],self.steps)
        
        #reduce parcelpoint capacity, if chosen
        if accepted:
            self.parcelPoints["parcelpoints"][idx-self.n_unique_customer_locs].remainingCapacity -= 1
        
        #info for plots and statistic
        info = self.steps,self.newCustomer.home.x,self.newCustomer.home.y,loc.x,loc.y,price
        
        #construct intermittent route kept in memory during booking horizon
        insertVeh,idx,costs = self.utils.cheapestInsertionRoute(loc,self.fleet)
        self.fleet["fleet"][insertVeh]["routePlan"].insert(idx,loc)
        
        #re-optimize the intermeittent route after X steps
        if self.steps % self.reopt_freq == 0:#do re-opt using HGS
            if self.load_data:
                self.data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix,self.data['id'])
            self.fleet,_ = self.utils.reopt_HGS(self.data)
        
        #generate new customer arrival and return state info
        self.curr_state = self.make_state()
        
        return self.curr_state.copy(), done, info, self.data