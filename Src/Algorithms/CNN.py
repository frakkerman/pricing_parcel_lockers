import numpy as np
import numpy.ma as ma
import torch
from torch import tensor, float32, int32
from math import sqrt
from Src.Utils.Utils import MemoryBuffer,readCVRPLIB,writeCVRPLIB,get_dist_mat_HGS,extract_route_HGS
from Src.Algorithms.Agent import Agent
from scipy.special import lambertw
from math import exp, e
from hygese import AlgorithmParameters, Solver

# This function implements the a Q-actor critic (QAC) algorithm
# contains the updates of actor and critic
class CNN(Agent):
    def __init__(self, config):
        super(CNN, self).__init__(config)
        
        
        
        
        
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
            
        # Define learning modules -- in LAR we have 3 learning modules as we have to additionally train the SL predictor
        self.modules = []
        
        self.historicRoutes = readCVRPLIB(self.config.paths['root'],config.veh_capacity,config.n_vehicles)
        self.dist_matrix = config.dist_matrix
        self.adjacency = config.adjacency
        self.first_parcelpoint_id = len(self.dist_matrix[0])-config.n_parcelpoints-1
        
        self.max_xcoord = max(config.coords, key=lambda x: x.x).x
        self.max_ycoord = max(config.coords, key=lambda y: y.y).y
        self.min_xcoord = min(config.coords, key=lambda x: x.x).x
        self.min_ycoord = min(config.coords, key=lambda y: y.y).y
        
        self.grid_dim = config.grid_dim
        
        self.memory =   MemoryBuffer(max_len=self.config.buffer_size, matrix_dim=int(self.grid_dim*self.grid_dim),
                                     target_dim=1, atype=int32, config=config)  
        
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
        theta = self.init_theta - (state[3] *  self.cool_theta)
        mltplr = self.cost_multiplier
        
        #cheapest insertion costs of every PP in current and historic routes
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data
        pp_costs = np.full(len(pps),1000000000.0)
        for pp in range(len(pps)):
            if state[2]["parcelpoints"][pp].remainingCapacity > 0:#check if parcelpont has remaining capacity
                pp_costs[pp] =  mltplr*((1-theta)*self.cheapestInsertionCosts(state[2]["parcelpoints"][pp].location, state[1]) + theta*self.historicCosts(state[2]["parcelpoints"][pp].location,self.historicRoutes))
        pp_sorted_args = state[2]["parcelpoints"][np.argpartition(pp_costs, self.k)[:self.k]]
        
        #get k best PPs
        action = self.get_id(pp_sorted_args)

        return action

    def get_action_pricing(self,state,training):
        #1 check if pp is feasible and obtain beta_0+beta_p, obtain costs per parcelpoint, obtain m
        theta = self.init_theta - (state[3] *  self.cool_theta)
        mltplr = self.cost_multiplier
        
        homeCosts = self.added_costs_home+mltplr*((1-theta)*(self.cheapestInsertionCosts(state[0].home, state[1]) ) + theta*(self.historicCosts(state[0].home,self.historicRoutes) ))
        sum_mnl = exp(state[0].home_util+(state[0].incentiveSensitivity*(homeCosts-self.revenue)))
        
        mask = ma.masked_array(state[2]["parcelpoints"], mask=self.adjacency[state[0].id_num])#only offer 20 closest
        pps = mask[mask.mask].data
        pp_costs= np.full((len(pps),1),1000000000.0)
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                util = self.mnl(state[0],pp)
                pp_costs[idx] = mltplr * ((1-theta)* ( self.cheapestInsertionCosts(pp.location, state[1]) )+ theta* ( self.historicCosts(pp.location,self.historicRoutes) ))
                sum_mnl += exp(util+(state[0].incentiveSensitivity*(pp_costs[idx]-self.revenue)))
       
        #2 obtain lambert w0
        lambertw0 = float(lambertw(sum_mnl/e)+1)/state[0].incentiveSensitivity
        
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
    
    def historicCosts(self,loc,fleets):
        costs = 0
        for f in fleets:
            costs += self.cheapestInsertionCosts(loc, f)
        return costs/len(fleets)
    
    def getdistance_euclidean(self,a,b):
        return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

    
    def mnl_euclid(self,customer,parcelpoint):
        distance = self.getdistance_euclidean(customer.home,parcelpoint.location)#distance from parcelpoint to home
        beta_p = exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
    
    def mnl_distmat(self,customer,parcelpoint):
        distance = self.dist_matrix[customer.id_num][parcelpoint.id_num]#distance from parcelpoint to home
        beta_p = exp(-distance/self.dist_scaler)
        return self.base_util + beta_p
    
    def update(self,data):
        #obtain final CVRP schedule after end of booking horizon
        if self.load_data:
            data["distance_matrix"] = get_dist_mat_HGS(self.dist_matrix,data['id'])
        fleet,cost = self.reopt_HGS_final(data)#do a final reopt
        
        #here you need to store the data
        #you need to store it in tensors
        #data contains:
            #a matrix like stratucture (spatial) of customers
            #customer divided in time when they arrived
            #the costs per customer (target
        #also you can use the update structure as used for LAR to update the model (initial_learning phase, load_data, etc.)
        
        
        features = self.get_matrix(fleet)
        target = self.get_per_customer_costs(fleet)
        
        self.memory.add(features,target)
        if not self.initial_phase:
            #simply update CNN after every new data point collected
                self.optimize()
                self.trajectory.reset()
        else:
            # action embeddings can be learnt offline
            # self.memory.add(s1, a1, a_emb1, r1, s2, int(done != 1))
            if self.memory.length >= self.config.buffer_size:
                self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)
        
        return cost
    
    def reopt_HGS_final(self,data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0#depot demand=0
        result = self.hgs_solver_final.solve_cvrp(data)  
        #update current routes
        fleet = extract_route_HGS(result,data)
        return fleet,result.cost

    def get_matrix(self,fleet):
        feature = [0]*int(self.grid_dim*self.grid_dim) 
        min_x = self.min_xcoord
        diff_x = self.max_xcoord-self.min_xcoord
        min_y = self.min_ycoord
        diff_y = self.max_ycoord-self.min_ycoord
        dim = self.grid_dim
        for v in fleet["fleet"]:
            for i in range(1,len(v["routePlan"])):
                column = int(dim*((v["routePlan"][i-1].x - min_x) / diff_x))
                row = int(dim*((v["routePlan"][i-1].y - min_y) / diff_y))
                feature[column+(row*dim)]+=1
        return feature
    
    def get_per_customer_costs(self,fleet):
        mltplr = self.cost_multiplier
        addedcosts_home = self.added_costs_home
        costs = []
        for v in fleet["fleet"]:
            for i in range(1,len(v["routePlan"])):
                #costs is composed of distance*mltplr
                costs.append( [mltplr * (0.5*self.dist_matrix[v["routePlan"][i-1].id_num][v["routePlan"][i].id_num] + 0.5*self.dist_matrix[v["routePlan"][i].id_num][v["routePlan"][i+1].id_num]),v["routePlan"][i].id_num] )
                if v["routePlan"][i].id_num < self.first_parcelpoint_id:
                    costs[-1][0] += addedcosts_home
        
        return costs

    def optimize(self):
        # Take one supervised step
        if not self.config.true_embeddings and self.config.emb_lambda > 0:# and self.memory.size >self.config.sup_batch_size:
            s1, a1, _, _, s2, _ = self.memory.sample(batch_size=self.config.sup_batch_size)
            self.self_supervised_update(s1, a1, s2, reg=self.config.emb_lambda)
    
    
    def self_supervised_update(self, s1, a1, s2, reg=1):
        self.clear_gradients()  # clear all the gradients from last run

        # If doing online updates, sharing the state features might be problematic!
        s1 = self.state_features.forward(s1)
        s2 = self.state_features.forward(s2)

        # ------------ optimize the embeddings ----------------
        loss_act_rep = self.action_rep.unsupervised_loss(s1, a1.view(-1), s2, normalized=True) * reg
        loss_act_rep.backward()

        # Directly call the optimizer's step fn to bypass lambda traces (if any)
        self.action_rep.optim.step()
        self.state_features.optim.step()

        return loss_act_rep.item()

    def initial_phase_training(self, max_epochs=-1):
        # change optimizer to Adam for unsupervised learning
        self.action_rep.optim = torch.optim.Adam(self.action_rep.parameters(), lr=1e-3)
        self.state_features.optim = torch.optim.Adam(self.state_features.parameters(), lr=1e-3)
        initial_losses = []

        print("Inital training phase started...")
        for counter in range(max_epochs):
            losses = []
            for s1, a1, _, _, s2, _ in self.memory.batch_sample(batch_size=self.config.sup_batch_size, randomize=True):
                loss = self.self_supervised_update(s1, a1, s2)
                losses.append(loss)

            initial_losses.append(np.mean(losses))
            if counter % 1 == 0:
                print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-10:])))
                if self.config.only_phase_one:
                    self.save()
                    print("Saved..")

            # Terminate initial phase once action representations have converged.
            if len(initial_losses) >= 20 and np.mean(initial_losses[-10:]) + 1e-5 >= np.mean(initial_losses[-20:]):
                print("Converged...")
                break

        # Reset the optim to whatever is there in config
        self.action_rep.optim = self.config.optim(self.action_rep.parameters(), lr=self.config.embed_lr)
        self.state_features.optim = self.config.optim(self.state_features.parameters(), lr=self.config.state_lr)

        print('... Initial training phase terminated!')
        self.initial_phase = False
        self.save()

        if self.config.only_phase_one:
            exit()
