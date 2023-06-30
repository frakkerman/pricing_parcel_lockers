import numpy as np
import numpy.ma as ma
from Src.Algorithms.Agent import Agent
from Src.Utils.Utils import get_dist_mat_HGS,writeCVRPLIB,extract_route_HGS
from hygese import AlgorithmParameters, Solver
from math import sqrt

# This function implements a baseline for pricing and offering decisions
class Baseline(Agent):
    def __init__(self, config):
        super(Baseline, self).__init__(config)
        
        #problem variant: pricing or offering
        if self.config.pricing:
            self.get_action = self.get_action_pricing
        else:
            self.get_action = self.get_action_offer
            self.k = config.k
        
        if config.load_data:
            self.get_dist = self.getdistance_distmat
        else:
            self.get_dist = self.getdistance_euclidean
            
        # Define learning modules -- in LAR we have 3 learning modules as we have to additionally train the SL predictor
        self.modules = []
        
        self.dist_matrix = config.dist_matrix
        self.adjacency = config.adjacency
        self.load_data = config.load_data

        self.home_price = config.price_home
        self.pp_price = config.price_pp
        
        #hgs settings
        ap_final = AlgorithmParameters(timeLimit=config.hgs_final_time)  # seconds
        self.hgs_solver_final = Solver(parameters=ap_final, verbose=False)#used for final route     
        self.filecounter=0
        
        #lambdas
        id_num = lambda x: x.id_num
        self.get_id = np.vectorize(id_num)

    def get_action_offer(self,state,training):   
        #check if pp is feasible
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data
        action = np.empty(0,dtype=int)
        dist_home = np.full((len(pps)),1000000000.0)
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                dist_home[idx] = self.get_dist(state[0],pp)
                
        pp_sorted_args = pps[np.argpartition(dist_home, self.k)[:self.k]]
        action = self.get_id(pp_sorted_args)

        return action
    
    def get_action_pricing(self,state,training):   
        #check if pp is feasible
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data        
        a_hat = np.zeros(len(pps)+1)
        a_hat[0] = self.home_price
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                a_hat[idx+1] = self.pp_price
        
        return np.around(a_hat,decimals=2)
    
    def update(self,data,state,done):
        if not done:
            return 0.0
        else:
            #obtain final CVRP schedule after end of booking horizon
            if self.load_data:
                data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix,data['id'])
            fleet,cost = self.reopt_HGS_final(data)#do a final reopt
            if self.config.save_routes:
                writeCVRPLIB(fleet,self.filecounter,self.config.paths['root'],int(self.config.n_vehicles*self.config.veh_capacity)-1,self.config.n_vehicles)            
                self.filecounter+=1
            return cost
    
    def reopt_HGS_final(self,data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0#depot demand=0
        result = self.hgs_solver_final.solve_cvrp(data)  
        #update current routes
        fleet = extract_route_HGS(result,data)
        return fleet,result.cost
    
    def getdistance_euclidean(self,a,b):
        return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
    
    def getdistance_distmat(self,a,b):
        return self.dist_matrix[a.id_num][b.id_num]