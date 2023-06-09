from math import exp
from numpy.random import gumbel
import numpy as np

class customerchoicemodel(object):
    def __init__(self,
                 base_util,
                 dist_scaler,
                 euclidean,
                 dist_mat):
        self.euclidean_distance = euclidean
        self.dist_scaler = dist_scaler
        self.base_util = base_util
        self.dist_mat = dist_mat
        if len(self.dist_mat)>0:
            self.mnl = self.mnl_distmat
        else:
            self.mnl = self.mnl_euclid
        
    def mnl_euclid(self,customer,parcelpoint):
        """
        multi-nomial logit model calculating euclidean distance
        """
        distance = self.euclidean_distance(customer.home,parcelpoint.location)#distance from parcelpoint to home
        beta_p = exp(-distance/self.dist_scaler)
        return self.base_util + beta_p

    def mnl_distmat(self,customer,parcelpoint):
        """
        multi-nomial logit model using distance matrix
        """
        distance = self.dist_mat[customer.id_num][parcelpoint.id_num]#distance from parcelpoint to home
        beta_p = exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
    
    def customerchoice_offer(self,customer,action,parcelpoints):
        """
        Customer choice model for the offering decision, i.e., action is 1 parcelpoint offer
        """
        pps = parcelpoints[action]
        shape = (len(action)+1, 1)
        utils= np.empty(shape)
        utils[0]=customer.home_util
        for idx,pp in enumerate(pps):
            utils[idx+1] = self.mnl(customer,pp)
        utils = np.add(utils,gumbel(0,1, np.shape(utils)))#mu=0,beta=1 (std Gumbel)
        
        idx = np.argmax(utils)
        if idx==0:
            return customer.home, False, -1, 0#home delivery
        else:
            return pps[idx-1].location, True, pps[idx-1].id_num,0#accept offer

    def customerchoice_pricing(self,customer,action,parcelpoints):
        """
        Customer choice model for the pricing decision, i.e., action is vector of prices for all PPs and home delivery
        """
        pps = parcelpoints[parcelpoints.mask].data
        shape = (len(pps)+1, 1)
        utils= np.empty(shape)
        utils[0]=customer.home_util+customer.incentiveSensitivity*action[0]
        for idx,pp in enumerate(pps):
            utils[idx+1] = self.mnl(customer,pp)
        utils = np.add(utils,customer.incentiveSensitivity*action.reshape((len(action),1)))#incentive
        utils = np.add(utils,gumbel(0,1, np.shape(utils)))#mu=0,beta=1 (std Gumbel)

        
        idx = np.argmax(utils)
        if idx==0:
            return customer.home, False, -1, action[0]#home delivery
        else:
            return pps[idx-1].location, True, pps[idx-1].id_num,action[idx-1]#accept offer