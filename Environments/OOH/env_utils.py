from math import sqrt
import numpy as np
from hygese import AlgorithmParameters, Solver

class utils_env(object):
    def __init__(self,
                 Location,
                 Vehicle,
                 Fleet,
                 ParcelPoint,
                 ParcelPoints,
                 vehicleCapacity,
                 num_vehicles,
                 pp_capacity,
                 data,
                 dist_matrix):
        
        self.Location = Location
        self.Vehicle = Vehicle
        self.Fleet = Fleet
        self.ParcelPoint = ParcelPoint
        self.ParcelPoints = ParcelPoints
        
        self.num_vehicles = num_vehicles
        self.vehicleCapacity = vehicleCapacity
        self.pp_capacity = pp_capacity
        
        self.dist_matrix = dist_matrix
        if len(dist_matrix)>0:
            self.addedcosts = self.addedcosts_distmat
        else:
            self.addedcosts = self.addedcosts_euclid
        
        # HGS Solvers initialization
        ap = AlgorithmParameters(timeLimit=3.2)  # seconds
        self.hgs_solver = Solver(parameters=ap, verbose=False)#used for intermittent route


    def getdistance_euclidean(self,a,b):
        return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
    
    def addedcosts_euclid(self,route,i,loc):
        costs = self.getdistance_euclidean(route[i-1],loc) + self.getdistance_euclidean(loc,route[i])\
                    - self.getdistance_euclidean(route[i-1],route[i])
        return costs
   
    def addedcosts_distmat(self,route,i,loc):
        costs = self.dist_matrix[route[i-1].id_num][loc.id_num] + self.dist_matrix[loc.id_num][route[i].id_num]\
                     - self.dist_matrix[route[i-1].id_num][route[i].id_num]
        return costs
    
    def cheapestInsertionRoute(self,newLoc,fleet):
        cheapestCosts = float("inf")
        bestLoc = -1
        bestVehicle = -1
        for v in fleet["fleet"]:#we do not check feasibility and let HGS deal with this
            for i in range(1,len(v["routePlan"])):
               addedCosts = self.addedcosts(v["routePlan"],i,newLoc)
               if addedCosts < cheapestCosts:
                   cheapestCosts = addedCosts
                   bestLoc = i
                   bestVehicle = v.id_num
        
        return bestVehicle,bestLoc,cheapestCosts
    
    def reopt_HGS(self,data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0#depot demand=0
        result = self.hgs_solver.solve_cvrp(data)
        #update current routes
        fleet = self.extract_route_HGS(result,data)
        return fleet
    
    def reset_fleet(self,fleet,initRouteplan):
        for v in fleet["fleet"]:
            v.routePlan = initRouteplan
            v.capacity = self.vehicleCapacity
        return fleet
    
    def get_parcelpoints(self):
        pps = np.empty(shape=(0,6))
        pp_locs = [self.Location(25,25,0,0),self.Location(25,75,1,0),self.Location(50,25,2,0),self.Location(50,75,3,0),self.Location(75,25,4,0),self.Location(75,75,5,0)]
        for p in range(len(pp_locs)):
            pps = np.append(pps,self.ParcelPoint(pp_locs[p],self.pp_capacity,p))
        return self.ParcelPoints(pps)
    
    def get_parcelpoints_from_data(self,data,start_id):
        pps = np.empty(shape=(0,len(data)))
        start_id=start_id
        for p in range(len(data)):
            pps = np.append(pps,self.ParcelPoint(data[p],self.pp_capacity,start_id))
            start_id+=1
        return self.ParcelPoints(pps)
    
    def reset_parcelpoints(self,parcelpoints):
        for p in parcelpoints["parcelpoints"]:
            p.remainingCapacity = self.pp_capacity
        return parcelpoints