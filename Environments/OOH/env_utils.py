from math import sqrt
import numpy as np
from hygese import AlgorithmParameters, Solver
from Src.Utils.Utils import extract_route_HGS

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
                 fraction_capacitated,
                 num_pps,
                 data,
                 dist_matrix,
                 hgs_time):
        
        self.Location = Location
        self.Vehicle = Vehicle
        self.Fleet = Fleet
        self.ParcelPoint = ParcelPoint
        self.ParcelPoints = ParcelPoints
        
        self.num_vehicles = num_vehicles
        self.vehicleCapacity = vehicleCapacity
        self.pp_capacity = pp_capacity
        self.fraction_capacitated = fraction_capacitated

        self.capacitated_pps = self.generate_fixed_list(num_pps, self.fraction_capacitated)

        self.dist_matrix = dist_matrix
        if len(dist_matrix)>0:
            self.addedcosts = self.addedcosts_distmat
        else:
            self.addedcosts = self.addedcosts_euclid
        
        # HGS Solvers initialization
        ap = AlgorithmParameters(timeLimit=hgs_time)  # seconds
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


    def generate_fixed_list(self, length, fraction_of_ones):
        # Set a fixed seed based on length and fraction to ensure consistency
        seed = hash((length, fraction_of_ones)) % (2**32)
        np.random.seed(seed)

        # Calculate number of ones
        number_of_ones = int(length * fraction_of_ones)

        # Create an array of zeros
        array = np.zeros(length, dtype=int)

        # Place ones at the first number_of_ones positions
        array[:number_of_ones] = 1

        # Shuffle the array to distribute ones randomly
        np.random.shuffle(array)

        return array.tolist()
    
    def reopt_HGS(self,data):
        data["demands"] = np.ones(len(data['x_coordinates']))
        data["demands"][0] = 0#depot demand=0
        result = self.hgs_solver.solve_cvrp(data)
        #update current routes
        fleet = extract_route_HGS(result,data)
        return fleet, result.cost
    
    def reset_fleet(self,fleet,initRouteplan):
        for v in fleet["fleet"]:
            v.routePlan = initRouteplan
            v.capacity = self.vehicleCapacity
        return fleet
    
    def get_parcelpoints(self):
        pps = np.empty(shape=(0,6))
        pp_locs = [self.Location(25,25,0,0),self.Location(25,75,1,0),self.Location(50,25,2,0),self.Location(50,75,3,0),
        self.Location(75,25,4,0),self.Location(75,75,5,0)]
        for p in range(len(pp_locs)):
            pps = np.append(pps,self.ParcelPoint(pp_locs[p],self.pp_capacity,p))
        return self.ParcelPoints(pps)
    
    def get_parcelpoints_from_data(self,data,start_id):
        pps = np.empty(shape=(0,len(data)))
        start_id=start_id
        for p in range(len(data)):
            if self.capacitated_pps[p] == 1:
                pps = np.append(pps,self.ParcelPoint(data[p], self.pp_capacity, start_id))
            else:
                pps = np.append(pps,self.ParcelPoint(data[p], 1000000, start_id))
            start_id += 1
        return self.ParcelPoints(pps)
    
    def reset_parcelpoints(self,parcelpoints):
        for i, p in enumerate(parcelpoints["parcelpoints"]):
            if self.capacitated_pps[i] == 1:
                p.remainingCapacity = self.pp_capacity
            else:
                p.remainingCapacity = 1000000
        return parcelpoints
