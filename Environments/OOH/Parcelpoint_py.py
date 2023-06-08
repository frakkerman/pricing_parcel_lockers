from __future__ import print_function

import numpy as np
import numpy.ma as ma
from Src.Utils.Utils import Space
from Environments.OOH.containers import Location,ParcelPoint,ParcelPoints,Vehicle,Fleet,Customer
from Environments.OOH.env_utils import utils_env
from Environments.OOH.customerchoice import customerchoicemodel
from Src.Utils.Utils import writeCVRPLIB, load_demand_data,read_adjacency


class Parcelpoint_py(object):
    def __init__(self,
                 model,
                 pricing = False,
                 n_vehicles=2,
                 veh_capacity=100,
                 parcelpoint_capacity=25,
                 incentive_sens=0.99,
                 base_util=0.2,
                 home_util=0.3,
                 reopt=2000,
                 saveRoutes = False,
                 path='Heuristic',
                 load_data=False,
                 city = 'Austin',
                 data_seed = 0,
                 num_offers = 1
                 ):
        
        #init fleet and parcelpoints
        self.n_vehicles = n_vehicles
        self.veh_capacity = veh_capacity
        self.pp_capacity = parcelpoint_capacity
        self.data = dict()
        
        #load data or generate data
        self.load_data = load_data
        if self.load_data:
            print("Note: the HGS python implementation (hygese 0.0.0.8) throws an assertion error for coords<0, you will need to outcomment this check in hygese.py")
            self.city = city
            self.data_seed = data_seed
            self.coords,self.dist_matrix,self.n_parcelpoints = load_demand_data(path,city,data_seed)
            self.utils = utils_env(Location,Vehicle,Fleet,ParcelPoint,ParcelPoints,self.veh_capacity,self.n_vehicles,self.pp_capacity,self.data,self.dist_matrix)
            self.depot = self.coords[0]
            self.parcelPoints = self.utils.get_parcelpoints_from_data(self.coords[-self.n_parcelpoints:])
            self.adjacency = read_adjacency(path,city,data_seed)
            self.action_space = Space(size=2**20)#we only select from the 20 closest parcelpoints
            self.get_customer = self.get_new_customer_from_data
            self.num_cust_loc = len(self.dist_matrix)-len(self.parcelPoints["parcelpoints"])-1
            self.dist_scaler = np.amax(self.dist_matrix)
        else:
            self.depot = Location(50,50,0)
            self.n_parcelpoints = 6
            self.dist_matrix = []
            self.utils = utils_env(Location,Vehicle,Fleet,ParcelPoint,ParcelPoints,self.veh_capacity,self.n_vehicles,self.pp_capacity,self.data,self.dist_matrix)
            self.parcelPoints = self.utils.get_parcelpoints()
            self.action_space = Space(size=2**self.n_parcelpoints)
            self.get_customer = self.generate_new_customer
            self.adjacency = np.ones(self.n_parcelpoints)
            self.dist_scaler = 10
        
        self.observation_space = Space(low=np.zeros(2, dtype=np.float32), high=np.full(2, 500, dtype=np.float32), dtype=np.float32)
                
        #customers
        self.home_util = home_util
        self.incentive_sens = incentive_sens
       
        self.newCustomer = Customer
        self.fleet = self.utils.get_fleet([self.depot,self.depot])

        #pricing of offering problem variant
        if pricing:
            self.action_space_matrix = self.get_actions(pricing,self.n_parcelpoints)
            self.customerchoice = customerchoicemodel(base_util,self.dist_scaler,self.utils.getdistance_euclidean,self.action_space_matrix,self.dist_matrix)
            self.customerChoice = self.customerchoice.customerchoice_pricing
            if self.load_data:
                self.n_actions = 20+1
            else:
                self.n_actions = self.n_parcelpoints+1
            self.get_delivery_loc = self.get_delivery_loc_pricing
        else:
            self.action_space_matrix = self.get_actions(pricing,self.n_parcelpoints)
            self.customerchoice = customerchoicemodel(base_util,self.dist_scaler,self.utils.getdistance_euclidean,self.action_space_matrix,self.dist_matrix)
            self.customerChoice = self.customerchoice.customerchoice_offer
            if self.load_data:
                self.n_actions = 20
            else:
                self.n_actions = self.n_parcelpoints
            self.get_delivery_loc = self.get_delivery_loc_offer
        
        self.steps = 0
        self.max_steps = (self.n_vehicles*self.veh_capacity)-1
        self.reopt_freq = reopt

        #state information required for policy
        if model=='Heuristic':
            self.make_state = self.make_state_full
        else:
            self.make_state = self.make_state_vector
        
        #save the final routes to use in the heuristic
        if saveRoutes:
            self.is_terminal = self.is_terminal_saveRoute
            self.filecounter = 0
            self.path = path
        else:
            self.is_terminal = self.is_terminal_orig
        self.reset()

    def seed(self, seed):
        self.seed = seed

    def reset(self,training=True):
        """
        Sets the environment to default conditions
        :return: None
        """
        self.fleet = self.utils.reset_fleet(self.fleet,[self.depot,self.depot])
        self.parcelPoints = self.utils.reset_parcelpoints(self.parcelPoints)
            
        self.steps = 0
        
        self.data['x_coordinates'] = self.depot.x
        self.data['y_coordinates'] =  self.depot.y
        self.data['id'] = 0
        self.data['vehicle_capacity'] = self.veh_capacity
        self.data['num_vehicles'] = self.n_vehicles
        
        self.curr_state = self.make_state()
        return self.curr_state

    def get_new_customer_from_data(self):
        idx = np.random.randint(1, self.num_cust_loc)
        home = self.coords[idx]#depot = 0
        return Customer(home,self.incentive_sens,self.home_util,idx)

    def generate_new_customer(self):
        home = Location(np.random.randint(0,500),np.random.randint(0,500),0)
        return Customer(home,self.incentive_sens,self.home_util,0)

    def make_state_vector(self):
        self.newCustomer = self.get_customer()
        state = [self.newCustomer.home.x,self.newCustomer.home.y]
        return state
    
    def make_state_full(self):
        self.newCustomer = self.get_customer()
        state = [self.newCustomer,self.fleet,self.parcelPoints,self.steps]
        return state
    
    def get_actions(self,pricing,n_parcelpoints):
        if pricing:
            return np.zeros((self.n_parcelpoints+1,1))
        else:
            return np.zeros((20,1))
        
    def is_terminal_orig(self):
        if self.steps >= self.max_steps:
            return 1
        else:
            return 0
    
    def is_terminal_saveRoute(self):
        if self.steps >= self.max_steps:
            if self.load_data:
                self.data["distance_matrix"] = self.utils.get_dist_mat_HGS(self.data['id'])
            fleet,cost = self.utils.reopt_HGS_final(self.data)#do a final reopt
            writeCVRPLIB(fleet,self.filecounter,self.path,self.max_steps,self.n_vehicles)
            self.filecounter += 1
            return 1
        else:
            return 0
    
    def get_delivery_loc_pricing(self,action):
        mask = ma.masked_array(self.parcelPoints["parcelpoints"], mask=self.adjacency[self.newCustomer.id_num])#only offer 20 closest
        return self.customerChoice(self.newCustomer,action,mask)
    
    def get_delivery_loc_offer(self,action):
        #get the chosen delivery location
        return self.customerChoice(self.newCustomer,action,self.parcelPoints["parcelpoints"])
    
    def step(self,action,training=True):
        done=False
        self.steps += 1
        reward = 1
        
        #get the customer's choice of delivery location
        loc,accepted,idx,price = self.get_delivery_loc(action)
        self.data['x_coordinates']= np.append(self.data['x_coordinates'],loc.x)
        self.data['y_coordinates'] = np.append(self.data['y_coordinates'],loc.y)
        self.data['id'] = np.append(self.data['id'],loc.id_num)
        
        #reduce parcelpoint capacity, if chosen
        if accepted:
            self.parcelPoints["parcelpoints"][idx].remainingCapacity -= 1
        
        #determine fianl CVRP schedule if we reached the end of the booking horizon
        cost=0.0
        if self.is_terminal():
            done=True
            if self.load_data:
                self.data["distance_matrix"] = self.utils.get_dist_mat_HGS(self.data['id'])
            fleet,cost = self.utils.reopt_HGS_final(self.data)#do a final reopt
        
        #info for plots and statistic
        info = self.steps,self.newCustomer.home.x,self.newCustomer.home.y,loc.x,loc.y,price,cost
        
        #construct intermittent route kept in memory during booking horizon
        insertVeh,idx,costs = self.utils.cheapestInsertionRoute(loc,self.fleet)
        self.fleet["fleet"][insertVeh]["routePlan"].insert(idx,loc)
        
        #re-optimize the intermeittent route after X steps
        if self.steps % self.reopt_freq == 0:#do re-opt using HGS
            if self.load_data:
                self.data["distance_matrix"] = self.utils.get_dist_mat_HGS(self.data['id'])
            self.fleet = self.utils.reopt_HGS(self.data)
        
        #generate new customer arrival and return state info
        self.curr_state = self.make_state()
        
        return self.curr_state.copy(), reward, done, info