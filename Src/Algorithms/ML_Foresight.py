import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
from torch import float32
from math import sqrt
from Src.Utils.Utils import MemoryBuffer,get_dist_mat_HGS,extract_route_HGS, get_matrix
from Src.Utils.Predictors import CNN
from Src.Algorithms.Agent import Agent
from scipy.special import lambertw
from math import exp, e
from hygese import AlgorithmParameters, Solver
from operator import itemgetter

# This function implements the a Q-actor critic (QAC) algorithm
# contains the updates of actor and critic
class ML_Foresight(Agent):
    def __init__(self, config):
        super(ML_Foresight, self).__init__(config)
               
        
        # heuristic parameters
        self.k = config.k
        self.init_theta = config.init_theta
        self.cool_theta = config.cool_theta#linear cooling scheme
        
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
        
        self.dist_matrix = config.dist_matrix
        self.adjacency = config.adjacency
        self.first_parcelpoint_id = len(self.dist_matrix[0])-config.n_parcelpoints-1
        
        
        self.grid_dim = config.grid_dim
        self.initial_phase = True
        self.memory =   MemoryBuffer(max_len=self.config.buffer_size, matrix_dim=self.grid_dim,
                                     target_dim=1, atype=float32, config=config)  
        
        self.customer_cell = get_matrix(config.coords,self.grid_dim)
        self.feat_idx = np.empty(0)
        self.features = np.empty((0,self.grid_dim*self.grid_dim))
        
        self.supervised_ml = CNN()
        self.optimizer = torch.optim.Adam(self.supervised_ml.parameters(),lr=1e-3)
        self.criterion = nn.MSELoss()
        
        #define learning modules
        self.modules = [('supervised_ml', self.supervised_ml)]
        
        self.load_data = config.load_data
        if self.load_data:
            self.addedcosts = self.addedcosts_distmat
            self.dist_scaler = np.amax(self.dist_matrix)
            self.mnl = self.mnl_distmat
        else:
            self.addedcosts = self.addedcosts_euclid
            self.dist_scaler = 10
            self.mnl = self.mnl_euclid
        
        #mnl parameters
        self.base_util = config.base_util
        self.cost_multiplier = (config.driver_wage+config.fuel_cost*config.truck_speed) / config.truck_speed
        self.added_costs_home = config.driver_wage*(config.del_time/60)
        self.revenue = config.revenue
        
        #hgs settings
        ap_final = AlgorithmParameters(timeLimit=3.2)  # seconds
        self.hgs_solver_final = Solver(parameters=ap_final, verbose=False)#used for final route        
        
        #lambdas
        id_num = lambda x: x.id_num
        self.get_id = np.vectorize(id_num)

    def get_action_offer(self,state,training):
        if self.initial_phase:
            return self.get_action_offerall(state,training)
        else:
        
            theta = self.init_theta - (state[3] *  self.cool_theta)
            mltplr = self.cost_multiplier
            
            #cheapest insertion costs of every PP in current and historic routes
            mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
            pps = mask[mask.mask].data
            pp_costs = np.full(len(pps),1000000000.0)
            
            #ML preds
            cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
            costs = self.get_prediction(cur_feat,state[0].home,pps)
            
            for pp in range(len(pps)):
                if state[2]["parcelpoints"][pp].remainingCapacity > 0:#check if parcelpont has remaining capacity               
                    pp_costs[pp] =  mltplr*((1-theta)*self.cheapestInsertionCosts(state[2]["parcelpoints"][pp].location, state[1]) + theta*(costs[pp+2]-costs[0]))
            pp_sorted_args = state[2]["parcelpoints"][np.argpartition(pp_costs, self.k)[:self.k]]
            
            #get k best PPs
            action = self.get_id(pp_sorted_args)

        return action

    def get_action_pricing(self,state,training):
        if self.initial_phase:
            a_hat = np.zeros(21)
            return np.around(a_hat,decimals=2)
        else:
            
            mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
            pps = mask[mask.mask].data
            pp_costs= np.full((len(pps),1),1000000000.0)
            
            #ML preds
            cur_feat = self.get_feature_rep_infer(state[1]["fleet"])
            costs = self.get_prediction(cur_feat,state[0].home,pps)
            
            #1 check if pp is feasible and obtain beta_0+beta_p, obtain costs per parcelpoint, obtain m
            theta = self.init_theta - (state[3] *  self.cool_theta)
            mltplr = self.cost_multiplier
            
            homeCosts = self.added_costs_home+mltplr*((1-theta)*(self.cheapestInsertionCosts(state[0].home, state[1]) ) + theta*( costs[1]-costs[0] ))
            sum_mnl = exp(state[0].home_util+(state[0].incentiveSensitivity*(homeCosts-self.revenue)))
            

            for idx,pp in enumerate(pps):
                if pp.remainingCapacity > 0:
                 #   ml_prediction = get feats do inferebce
                    
                    util = self.mnl(state[0],pp)
                    pp_costs[idx] = mltplr * ((1-theta)* ( self.cheapestInsertionCosts(pp.location, state[1]) )+ theta*(costs[idx+2]-costs[0]) )
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
    
    def get_action_offerall(self,state,training):   
        #check if pp is feasible
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data
        action = np.empty(0,dtype=int)
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                action = np.append(action,pp.id_num)
        return action
    
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
    
    def getdistance_euclidean(self,a,b):
        return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

    def get_prediction(self,cur_feat,home,pps):
        new_feat = torch.cat((2+len(pps))*[cur_feat])
        new_feat[1][self.customer_cell[home.id_num][0]][self.customer_cell[home.id_num][1]]+=1
        for idx,p in enumerate(pps):
            new_feat[idx+2][self.customer_cell[p.location.id_num][0]][self.customer_cell[p.location.id_num][1]]+=1
        
        costs = []
        for feat  in new_feat:
            costs.append(self.supervised_ml(feat.unsqueeze(0)).item())
        return costs
                                                       

    def mnl_euclid(self,customer,parcelpoint):
        distance = self.getdistance_euclidean(customer.home,parcelpoint.location)#distance from parcelpoint to home
        beta_p = exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
    
    def mnl_distmat(self,customer,parcelpoint):
        distance = self.dist_matrix[customer.id_num][parcelpoint.id_num]#distance from parcelpoint to home
        beta_p = exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
    
    
    def update(self,data,state,done=False):
        #first obtain data      
        if not done:
            self.features = np.vstack(( self.features, self.get_feature_rep(data).flatten()))
            return 0.0
        else:
            #obtain final CVRP schedule after end of booking horizon
            if self.load_data:
                data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix,data['id'])
            fleet,cost = self.reopt_HGS_final(data)#do a final reopt
            
            target = self.get_per_customer_costs(fleet)
            target = sorted(target, key=itemgetter(0))#sort in order of arrival (same as features)
            self.memory.add(self.features,target)
            
            self.features = np.empty((0,self.grid_dim*self.grid_dim))

            #optionally update model
            if self.initial_phase:#train model initial phase
                if self.memory.length >= self.config.buffer_size:
                    self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)
            elif not self.config.only_phase_one:
                #simply update CNN after every new data point collected
                    self.optimize()
        
            return cost

    def optimize(self):
        # Take one supervised step
        feat,target = self.memory.sample(batch_size=self.config.batch_size)
        self.self_supervised_update(feat,target)
    
    
    def self_supervised_update(self, feat,target):
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.supervised_ml(feat)
        loss = self.criterion(outputs, target[0])
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def initial_phase_training(self, max_epochs=-1):
        initial_losses = []

        print("Inital training phase started...")
        for counter in range(max_epochs):
            losses = []
            for feat,target in self.memory.batch_sample(batch_size=self.config.batch_size, randomize=True):
                loss = self.self_supervised_update(feat,target)
                losses.append(loss)

            initial_losses.append(np.mean(losses))
            if counter % 1 == 0:
                print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-10:])))
                if self.config.only_phase_one:
                    self.memory.save(self.config.paths['checkpoint']+'initial_' )
                    self.save()
                    print("Saved..")

            # Terminate initial phase once it have converged.
            if len(initial_losses) >= 20 and np.mean(initial_losses[-10:]) + 1e-5 >= np.mean(initial_losses[-20:]):
                print("Converged...")
                break
        print('... Initial training phase terminated!')
        self.initial_phase = False
        self.memory.save(self.config.paths['checkpoint']+'initial_' )
        self.save()
        
    def get_feature_rep(self,data):
        feature = np.zeros((self.grid_dim,self.grid_dim))
        for i in data["id"]:
            feature[self.customer_cell[i][0]][self.customer_cell[i][1]]+=1
        return feature
    
    def get_feature_rep_infer(self,fleet):
        feature = np.zeros((self.grid_dim,self.grid_dim))
        for v in fleet:
            for i in v["routePlan"]:
                feature[self.customer_cell[i.id_num][0]][self.customer_cell[i.id_num][1]]+=1
        return torch.tensor(feature,dtype=float32,requires_grad=False).unsqueeze(0)
    

    
    def reopt_HGS_final(self,data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0#depot demand=0
        result = self.hgs_solver_final.solve_cvrp(data)  
        #update current routes
        fleet = extract_route_HGS(result,data)
        return fleet,result.cost
    
    def get_per_customer_costs(self,fleet):
        mltplr = self.cost_multiplier
        addedcosts_home = self.added_costs_home
        costs = []
        for v in fleet["fleet"]:
            for i in range(0,len(v["routePlan"])):
                #costs is composed of distance*mltplr
                if i==0:
                    costs.append( [v["routePlan"][i].time,mltplr * (0.5*self.dist_matrix[0][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][v["routePlan"][i+1].id_num])] )
                elif i==len(v["routePlan"])-1:
                    costs.append( [v["routePlan"][i].time,mltplr * (0.5*self.dist_matrix[v["routePlan"][i-1].id_num][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][0])] )
                else:
                    costs.append( [v["routePlan"][i].time,mltplr * (0.5*self.dist_matrix[v["routePlan"][i-1].id_num][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][v["routePlan"][i+1].id_num])] )
                
                if v["routePlan"][i].id_num < self.first_parcelpoint_id:
                    costs[-1][0] += addedcosts_home
        
        return costs