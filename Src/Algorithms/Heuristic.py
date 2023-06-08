import numpy as np
import numpy.ma as ma
from math import sqrt
from Src.Utils.Utils import readCVRPLIB,load_demand_data,read_adjacency
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis
from scipy.special import lambertw
from math import exp, e

# This function implements the a Q-actor critic (QAC) algorithm
# contains the updates of actor and critic
class Heuristic(Agent):
    def __init__(self, config):
        super(Heuristic, self).__init__(config)

        # Obtain state features
        self.state_features = Basis.get_Basis(config=config)

        # Initialize action space matrix
        self.action_space_matrix = config.env.action_space_matrix
        
        # heuristic parameters
        self.k = config.k
        self.init_theta = config.init_theta
        if config.cool_theta:
            self.cool_theta = self.init_theta / (config.n_vehicles*config.veh_capacity)#linear cooling scheme
        else:
            self.cool_theta= 0.0
        
        #problem variant: pricing or offering
        if self.config.pricing:
            if config.offer_all:
                raise ValueError("Offer all heuristic not available for pricing problem variant" )
            self.get_action = self.get_action_pricing
            self.max_p = config.max_price
            self.min_p = config.min_price
        else:
            if config.offer_all:
                self.get_action = self.get_action_offerall
            else:
                self.get_action = self.get_action_offer
            
        # Define learning modules -- in LAR we have 3 learning modules as we have to additionally train the SL predictor
        self.modules = []
        
        self.historicRoutes = readCVRPLIB(self.config.paths['root'],config.veh_capacity,config.n_vehicles)
        if config.load_data:
            _,self.dist_matrix,_ = load_demand_data(self.config.paths['root'],config.city,config.data_seed)
            self.addedcosts = self.addedcosts_distmat
            self.n_pp = 20 #we only offer the 20 closest pps
            self.adjacency = read_adjacency(self.config.paths['root'],config.city,config.data_seed)
            self.dist_scaler = np.amax(self.dist_matrix)
        else:
            self.dist_matrix = []
            self.addedcosts = self.addedcosts_euclid
            self.n_pp = 6
            self.adjacency = np.ones(self.n_pp)
            self.dist_scaler = 10
        
        self.shape = (self.n_pp, 1)
        
        
        # pricing related mnl parameters
        if len(self.dist_matrix)>0:
            self.mnl = self.mnl_distmat
        else:
            self.mnl = self.mnl_euclid
        self.base_util = config.base_util
        self.cost_multiplier = (config.driver_wage+config.fuel_cost*config.truck_speed) / config.truck_speed
        self.added_costs_home = config.driver_wage*(config.del_time/60)
        self.revenue = config.revenue
        
        #lambdas
        id_num = lambda x: x.id_num
        self.get_id = np.vectorize(id_num)

    def get_action_offer(self,state,training):
        theta = self.init_theta - (state[3] *  self.cool_theta)
        mltplr = self.cost_multiplier
        
        #cheapest insertion costs of every PP in current and historic routes
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data
        pp_costs = np.full(len(pps),1000000000.0)
        for pp in range(self.n_pp):
            if state[2]["parcelpoints"][pp].remainingCapacity > 0:#check if parcelpont has remaining capacity
                pp_costs[pp] =  mltplr*((1-theta)*self.cheapestInsertionCosts(state[2]["parcelpoints"][pp].location, state[1]) + theta*self.historicCosts(state[2]["parcelpoints"][pp].location,self.historicRoutes))
        pp_sorted_args = state[2]["parcelpoints"][np.argpartition(pp_costs, self.k)[:self.k]]
        
        #get k best PPs
        action = self.get_id(pp_sorted_args)

        return action, action

    def get_action_pricing(self,state,training):
        #1 check if pp is feasible and obtain beta_0+beta_p, obtain costs per parcelpoint, obtain m
        theta = self.init_theta - (state[3] *  self.cool_theta)
        mltplr = self.cost_multiplier
        
        homeCosts = self.added_costs_home+mltplr*((1-theta)*(self.cheapestInsertionCosts(state[0].home, state[1]) ) + theta*(self.historicCosts(state[0].home,self.historicRoutes) ))
        sum_mnl = exp(state[0].home_util+(state[0].incentiveSensitivity*(homeCosts-self.revenue)))
        
        pp_costs= np.full(self.shape,1000000000.0)
        
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                util = self.mnl(state[0],pp)
                pp_costs[idx] = mltplr * ((1-theta)* ( self.cheapestInsertionCosts(pp.location, state[1]) )+ theta* ( self.historicCosts(pp.location,self.historicRoutes) ))
                sum_mnl += exp(util+(state[0].incentiveSensitivity*(pp_costs[idx]-self.revenue)))
       
        #2 obtain lambert w0
        lambertw0 = float(lambertw(sum_mnl/e)+1)/state[0].incentiveSensitivity
        
        # 3 calculate discounts/prices
        a_hat = np.zeros(self.n_pp+1)
        a_hat[0] = homeCosts - self.revenue - lambertw0
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                a_hat[idx+1] = pp_costs[idx] - self.revenue - lambertw0
        
        a_hat = np.clip(a_hat,self.min_p,self.max_p)
        return np.around(a_hat,decimals=2),a_hat
    
    def get_action_offerall(self,state,training):   
        #check if pp is feasible
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data
        action = np.empty(0)
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                action = np.append(action,pp.id_num)
        return action,action
    
    def addedcosts_euclid(self,route,i,loc):
        costs = self.getdistance_euclidean(route[i-1],loc) + self.getdistance_euclidean(loc,route[i])\
                    - self.getdistance_euclidean(route[i-1],route[i])
        return costs/self.dist_scaler
   
    def addedcosts_distmat(self,route,i,loc):
        costs = self.dist_matrix[route[i-1].id_num][loc.id_num] + self.dist_matrix[loc.id_num][route[i].id_num]\
                     - self.dist_matrix[route[i-1].id_num][route[i].id_num]
        return costs/self.dist_scaler     
                    
    def cheapestInsertionCosts(self,loc,fleet):
        cheapestCosts = float("inf")
        for v in fleet["fleet"]:#note we do not check feasibility of insertion here, let this to HGS
            for i in range(1,len(v["routePlan"])):
               addedCosts = self.addedcosts(v["routePlan"],i,loc)
               if addedCosts < cheapestCosts:
                   cheapestCosts = addedCosts
        
        return cheapestCosts
    
    def historicCosts(self,loc,fleets):
        costs = 0
        for f in fleets:
            costs += self.cheapestInsertionCosts(loc, f)
        return costs/len(fleets)
    
    def getdistance_euclidean(self,a,b):
        return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
    
    def get_best_match(self, action):
        action = np.where(np.all(self.action_space_matrix==action,axis=1))[0][0]
        return action
    
    def mnl_euclid(self,customer,parcelpoint):
        #multi-nomial logit model
        distance = self.getdistance_euclidean(customer.home,parcelpoint.location)#distance from parcelpoint to home
        beta_p = exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
    
    def mnl_distmat(self,customer,parcelpoint):
        """
        multi-nomial logit model using distance matrix
        """
        distance = self.dist_matrix[customer.id_num][parcelpoint.id_num]#distance from parcelpoint to home
        beta_p = exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
   
    
    
    def update(self, s1, a1, a_hat_1, r1, s2, done):
        loss_actor = 0
        loss_critic = 0
        return loss_actor,loss_critic