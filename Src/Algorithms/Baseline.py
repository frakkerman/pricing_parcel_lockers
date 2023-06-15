import numpy as np
import numpy.ma as ma
from Src.Algorithms.Agent import Agent

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

        self.home_price = config.price_home
        self.pp_price = config.price_pp

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