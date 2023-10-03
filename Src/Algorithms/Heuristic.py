import numpy as np
import numpy.ma as ma
from Src.Utils.Utils import readCVRPLIB,get_dist_mat_HGS
from Src.Algorithms.Agent import Agent
from scipy.special import lambertw
from math import exp, e, sqrt
from hygese import AlgorithmParameters, Solver

# This function implements the method proposed in Yang et. al (2016) adapted to the OOH setting
#"Choice-Based Demand Management and Vehicle Routing in E-Fulfillment"
class Heuristic(Agent):
    def __init__(self, config):
        super(Heuristic, self).__init__(config)
        
        # heuristic parameters
        self.k = config.k
        self.init_theta = config.init_theta
        self.cool_theta = config.cool_theta#linear cooling scheme
        
        #problem variant: pricing or offering
        if self.config.pricing:
            self.get_action = self.get_action_pricing
            self.max_p = config.max_price
            self.min_p = config.min_price
        else:
            self.get_action = self.get_action_offer
            
        # Define learning modules -- in LAR we have 3 learning modules as we have to additionally train the SL predictor
        self.modules = []
        
        self.historicRoutes = readCVRPLIB(self.config.paths['root'],config.veh_capacity,config.n_vehicles)
        self.dist_matrix = config.dist_matrix
        self.adjacency = config.adjacency
        self.load_data = config.load_data
        if self.load_data:
            self.addedcosts = self.addedcosts_distmat
            self.dist_scaler = 1#np.amax(self.dist_matrix)
            self.mnl = self.mnl_distmat
        else:
            self.addedcosts = self.addedcosts_euclid
            self.dist_scaler = 1# 10
            self.mnl = self.mnl_euclid
        
        #mnl parameters
        self.base_util = config.base_util
        self.cost_multiplier = (config.driver_wage+config.fuel_cost*config.truck_speed) / 3600
        self.wage = config.driver_wage
       # self.added_costs_home = config.driver_wage*(config.del_time/60)
        self.revenue = config.revenue
        
        #hgs settings
        ap_final = AlgorithmParameters(timeLimit=config.hgs_final_time)  # seconds
        self.hgs_solver_final = Solver(parameters=ap_final, verbose=False)#used for final route        
        
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
        for pp in range(len(pps)):
            if state[2]["parcelpoints"][pp].remainingCapacity > 0:#check if parcelpont has remaining capacity
                pp_costs[pp] =  mltplr*((1-theta)*self.cheapestInsertionCosts(state[2]["parcelpoints"][pp].location, state[1]) + theta*self.historicCosts(state[2]["parcelpoints"][pp].location,self.historicRoutes))
        pp_sorted_args = pps[np.argpartition(pp_costs, self.k)[:self.k]]
        
        #get k best PPs
        action = self.get_id(pp_sorted_args)

        return action

    def get_action_pricing(self,state,training):
        #1 check if pp is feasible and obtain beta_0+beta_p, obtain costs per parcelpoint, obtain m
        theta = self.init_theta - (state[3] *  self.cool_theta)
        mltplr = self.cost_multiplier
        
        homeCosts = state[0].service_time/60*self.wage+mltplr*((1-theta)*(self.cheapestInsertionCosts(state[0].home, state[1]) ) + theta*(self.historicCosts(state[0].home,self.historicRoutes) ))
        sum_mnl = exp(self.base_util+state[0].home_util+(state[0].incentiveSensitivity*(homeCosts-self.revenue)))
        
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data
        pp_costs= np.full((len(pps),1),1000000000.0)
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                util = self.mnl(state[0],pp)
                pp_costs[idx] = mltplr * ((1-theta)* ( self.cheapestInsertionCosts(pp.location, state[1]) )+ theta* ( self.historicCosts(pp.location,self.historicRoutes) ))
                sum_mnl += exp(util+(state[0].incentiveSensitivity*(pp_costs[idx]-self.revenue)))
       
        #2 obtain lambert w0
        lambertw0 = (lambertw(sum_mnl/e).real+1)/state[0].incentiveSensitivity
        
        # 3 calculate discounts/prices
        a_hat = np.zeros(len(pps)+1)
        a_hat[0] = homeCosts - self.revenue - lambertw0
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                a_hat[idx+1] = pp_costs[idx] - self.revenue - lambertw0
        
        a_hat = np.clip(a_hat,self.min_p,self.max_p)
        return np.around(a_hat,decimals=2)
    
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
        for v in fleet["fleet"]:#note we do not check feasibility of insertion here, leave this to HGS
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

    
    def mnl_euclid(self,customer,parcelpoint):
        distance = self.getdistance_euclidean(customer.home,parcelpoint.location)#distance from parcelpoint to home
        beta_p = -exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
    
    def mnl_distmat(self,customer,parcelpoint):
        distance = self.dist_matrix[customer.id_num][parcelpoint.id_num]#distance from parcelpoint to home
        beta_p = -exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
    
    def update(self,data,state,done):
        if not done:
            return 0.0
        else:
            #obtain final CVRP schedule after end of booking horizon
            if self.load_data:
                data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix,data['id'])
            cost = self.reopt_HGS_final(data)#do a final reopt
            return cost
    
    def reopt_HGS_final(self,data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0#depot demand=0
        result = self.hgs_solver_final.solve_cvrp(data)  
        #update current routes
        return result.cost