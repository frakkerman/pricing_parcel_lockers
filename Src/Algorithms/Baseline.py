import numpy as np
import numpy.ma as ma
from Src.Algorithms.Agent import Agent
from Src.Utils.Utils import get_dist_mat_HGS
from hygese import AlgorithmParameters, Solver

# This function implements a baseline for pricing and offering decisions
class Baseline(Agent):
    def __init__(self, config):
        super(Baseline, self).__init__(config)
        
        #problem variant: pricing or offering
        if self.config.pricing:
            self.get_action = self.get_action_pricing
        else:
            self.get_action = self.get_action_offerall
            
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

    def get_action_offerall(self,state,training):   
        #check if pp is feasible
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data
        action = np.empty(0,dtype=int)
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                action = np.append(action,pp.id_num)
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
            cost = self.reopt_HGS_final(data)#do a final reopt
            return cost
    
    def reopt_HGS_final(self,data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0#depot demand=0
        result = self.hgs_solver_final.solve_cvrp(data)  
        return result.cost