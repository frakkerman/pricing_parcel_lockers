from __future__ import print_function
import numpy as np
import torch
from torch import float32
import shutil
import matplotlib.pyplot as plt
from os import path, mkdir, listdir, fsync, name
import importlib
from time import time
import sys
from Environments.OOH.containers import Location,Vehicle,Fleet
from math import trunc, sqrt
import hygese

np.random.seed(0)
torch.manual_seed(0)
dtype = torch.FloatTensor

class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, method): # restore
        self.terminal = sys.stdout
        self.file = 'file' in method
        self.term = 'term' in method
        self.log_path = log_path
        self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        self.log.flush()
        fsync(self.log.fileno())
   

def save_plots_test_runs(test_returns,test_std,step_time,config):
    np.save(config.paths['results'] + "eval_dist_mean", test_returns)
    np.save(config.paths['results'] + "eval_dist_std", test_std)
    np.save(config.paths['results'] + "eval_step_time", step_time)
    x = config.save_after * np.arange(0, len(test_returns))
    plt.figure()
    plt.ylabel("Total HGS time")
    plt.xlabel("Episode")
    plt.title("Performance")
    plt.plot(x,test_returns,color='#CC4F1B')
    plt.fill_between(x,np.array(test_returns)+np.array(test_std),np.array(test_returns)-np.array(test_std),alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.savefig(config.paths['results'] + "hgs_time_eval_runs.png")
    plt.close()
    
    plt.figure()
    plt.ylabel("Step time")
    plt.title("Performance")
    plt.boxplot(step_time)
    plt.savefig(config.paths['results'] + "box_eval_step_time.png")
    plt.close()

def save_plots_stats(num_cust,stats,costs,run_time,actions,config,episode):
    info = np.array(stats)
    np.save(config.paths['results'] + "run_time", run_time)
    
    pricing_revenue = 0
    if config.pricing:
        #1 line chart of discounts over time
        given_price = np.split(info[:,-1],episode+1,axis=0)
        mean_price = np.mean(given_price, axis=0)
        std_price = np.std(given_price, axis=0)
        x = np.arange(0, len(mean_price))
        plt.figure()
        plt.ylabel("Given price")
        plt.xlabel("Customer arrival")
        plt.title("Given price")
        plt.plot(x,mean_price,color='#CC4F1B')
        plt.fill_between(x,mean_price+std_price,mean_price-std_price,alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.savefig(config.paths['results'] + "accepted_price.png")
        plt.close()
        
        #2 boxplot of price discounts
        all_prices = np.concatenate( actions, axis=0 ) 
        given_price = np.concatenate( given_price, axis=0 )
        pricing_revenue = sum(given_price)/(episode+1)
        plt.figure()
        plt.ylabel("Price")
        plt.title("Performance")
        plt.boxplot([all_prices,given_price])
        plt.xticks([1, 2], ['Given price', 'Accepted price'])
        plt.savefig(config.paths['results'] + "box_prices.png")
        plt.close()
        
        #save data
        np.save(config.paths['results'] + "given_price", all_prices)
        np.save(config.paths['results'] + "accepted_price", given_price)
    
    #3 barchart home/pp deliveries
    count_home = np.count_nonzero(np.logical_and(info[:,1]==info[:,3], info[:,2]==info[:,4]))
    plt.figure()
    plt.ylabel("Percentage")
    plt.title("Performance")
    plt.bar([1,2],[count_home/len(info),1-(count_home/len(info))])
    plt.xticks([1, 2], ['Home deliveries', 'Parcel point deliveries'])
    plt.ylim((0.0,1.0))
    plt.savefig(config.paths['results'] + "bar_deliveries.png")
    plt.close()
    
    print(count_home/len(info))
    
    #4 boxplot final hgs distance
    costs =  np.array([i for i in costs if i != 0])
    plt.figure()
    plt.ylabel("Total time")
    plt.title("Performance")
    plt.boxplot(costs)
    plt.savefig(config.paths['results'] + "box_hgs_time.png")
    plt.close()
    
    #5 count number of unique parcelpoints
    count_pps = len(np.unique(info[np.logical_or(info[:,1]!=info[:,3], info[:,2]!=info[:,4])][:,3]))
    np.save(config.paths['results'] + "num_active_parcelpoints", count_pps)
    
    #6 bar times
    #TODO: add the actua; servcie tie (already adaded to stats, only need to abstract it)
    added_costs_home = (config.del_time/60)
    drive_time = np.mean(costs)
    service_time = added_costs_home*count_home
    plt.figure()
    plt.ylabel("Hours")
    plt.title("Performance")
    plt.bar([1,2],[drive_time,service_time])
    plt.xticks([1, 2], ['Driving time', 'Service time'])
    plt.savefig(config.paths['results'] + "bar_drive_del_time.png")
    plt.close()
    
    #TODO: add walkawy count and costs
    
    #7cost and revenue
    cost_multiplier = (config.driver_wage+config.fuel_cost*config.truck_speed) / 3600
    revenue = (len(info)/(episode+1))*config.revenue
    total_costs = added_costs_home*count_home+(np.mean(costs)*cost_multiplier)
    total_costs += count_home*config.home_failure*config.failure_cost#costs of failed delivery
    plt.figure()
    plt.ylabel("Monetary unit")
    plt.title("Performance")
    plt.bar([1,2,3],[revenue,pricing_revenue,total_costs])
    plt.xticks([1, 2,3], ['Revenue', 'Pricing', 'Total costs'])
    plt.savefig(config.paths['results'] + "bar_revenue_cost.png")
    plt.close()
       
    #maybe add stats later considering OOH capacity
    
    #save data
    np.save(config.paths['results'] + "drive_time", drive_time)
    np.save(config.paths['results'] + "service_time", service_time)
    np.save(config.paths['results'] + "pricing_revenue", pricing_revenue)
    np.save(config.paths['results'] + "revenues", revenue)
    np.save(config.paths['results'] + "total_costs", total_costs)
    np.save(config.paths['results'] + "hgs_times", costs)
    np.save(config.paths['results'] + "percent_home_deliveries", count_home/len(info))


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def search(dir, name, exact=False):
    all_files = listdir(dir)
    for file in all_files:
        if exact and name == file:
            return path.join(dir, name)
        if not exact and name in file:
            return path.join(dir, name)
    else:
        # recursive scan
        for file in all_files:
            if file == 'Experiments':
                continue
            _path = path.join(dir, file)
            if path.isdir(_path):
                location = search(_path, name, exact)
                if location:
                    return location

def dynamic_load(dir, name, load_class=False):
    try:
        abs_path = search(dir, name).split('/')[1:]

        if len(abs_path) == 0:
            abs_path = search(dir, name).split('\\')[1:]
        pos = abs_path.index('ooh_code')

        module_path = '.'.join([str(item) for item in abs_path[pos + 1:]])
        print("Module path: ", module_path, name)
        if load_class:
            obj = getattr(importlib.import_module(module_path), name)
        else:
            obj = importlib.import_module(module_path)
        print("Dynamically loaded from: ", obj)
        return obj
    except:
        raise ValueError("Failed to dynamically load the class: " + name )

def check_n_create(dir_path, overwrite=False):
    try:
        if not path.exists(dir_path):
            mkdir(dir_path)
        else:
            if overwrite:
               shutil.rmtree(dir_path)
               mkdir(dir_path)
    except FileExistsError:
        print("\n ##### Warning File Exists... perhaps multi-threading error? \n")

def create_directory_tree(dir_path):
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    dir_path = str.split(dir_path, sep=sepa)[1:-1]  #Ignore the blank characters in the start and end of string
    for i in range(len(dir_path)):
        check_n_create(path.join(sepa, *(dir_path[:i + 1])))


def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)


def writeCVRPLIB(fleet,filename,pathh,n_cust,n_veh):
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    pathh = pathh + sepa+'Src'+ sepa+'Algorithms'+sepa+'CVRPLIB'+sepa
    folder = str(n_cust+1)+'_'+str(n_veh)+sepa
    if filename==0:
        if not path.exists(pathh+folder):
            create_directory_tree(pathh+folder)
        # else:
        #     while input("Folder exists already, possibly overwriting existing files, do you want to continue? [y/n]") == "n":
        #         exit('Exit program, folder exists')
    route=[]
    for v in range(len(fleet["fleet"])):
        xy=[]
        for i in fleet["fleet"][v]["routePlan"]:
            xy.append(str(i.x)+'\t'+str(i.y)+'\t'+str(i.id_num))
        route.append(xy)
    with open(pathh+folder+'CVRPLIB'+str(filename)+'.txt', 'w') as fp:
        for v in range(len(fleet["fleet"])):
            fp.write("Route_"+str(v)+'\n')
            fp.write('\n'.join(route[v]))
            fp.write('\n')

def readCVRPLIB(pathh,v_cap,n_veh):
    historicRoutes = np.empty(0)
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    pathh = pathh+sepa+'Src'+sepa+'Algorithms'+sepa+'CVRPLIB'+sepa+str(v_cap*n_veh)+'_'+str(n_veh)
    if path.exists(pathh):
        for filename in listdir(pathh):
            f = path.join(pathh, filename)
            # checking if it is a file
            if path.isfile(f):
                file = open(f, "r")
                routeplans=[ [] for i in range(n_veh)]
                idx = -1
                for i in file:
                    if not i.startswith('Route'):
                        loc = i.strip().split('\t')
                        loc = Location(float(loc[0]),float(loc[1]),int(loc[2]),0)#we do not care about time here
                        routeplans[idx].append(loc)
                    else:
                        idx +=1
                vehicles=[]
                for v in range(n_veh):
                    vehicles.append(Vehicle(routeplans[v],v_cap,v))
                historicRoutes = np.append(historicRoutes,Fleet(vehicles))
        return historicRoutes
    else:
        raise ValueError("Failed to load the historic routes: " + str(v_cap*n_veh)+'_'+str(n_veh) )

def sixhump_func(x,y):
    """
    for documentation and visualisation of the sixhump camelback function, see: https://www.sfu.ca/~ssurjano/camel6.html
    """
    return (4-2.1*x**2+(x**4/3))*x**2+x*y+(-4+4*y**2)*x**2+6

def calculate_service_time(coords):
    """
    We project the coordinates onto the domain [-3,3]x[-2,2] and next calculate service times using the 6-hump camel function
    """
    max_xcoord = max(coords, key=lambda x: x.x).x
    max_ycoord = max(coords, key=lambda y: y.y).y
    min_xcoord = min(coords, key=lambda x: x.x).x
    min_ycoord = min(coords, key=lambda y: y.y).y
    diff_x = max_xcoord-min_xcoord
    diff_y = max_ycoord-min_ycoord
    
    mult_x = 6
    mult_y = 4
    #standardize x-min_x/diff, we use domain: x-[-3,3] y-[-2,2]
    service_times = np.zeros([0])
    for coord in coords:
        x1 = (((coord.x-min_xcoord)/diff_x)*mult_x)-3
        y1 = (((coord.y-min_ycoord)/diff_y)*mult_y)-2
        sixhump = np.around(np.clip(sixhump_func(x1,y1),1,10),decimals=2)
        service_times = np.append(service_times,sixhump)
        
    return service_times

def getdistance_euclidean(a,b):
    return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def load_demand_data(pathh,instance,data_seed):
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    if instance=='Austin' or instance=='Seattle':
        instance_folder = 'Amazon_data'
        instance_size = '_700_'
    else:
        instance_folder = 'HombergerGehring'
        instance_size = '_90_'
    
    pathh = pathh+sepa+'Environments'+sepa+'OOH'+sepa+instance_folder+sepa+instance+sepa
    if path.exists(pathh):
        pathh = pathh+instance+instance_size+str(data_seed)
        f = pathh+"_coords.txt"
        if path.isfile(f):
            file = open(f, "r")
            coords = np.zeros([0])
            for j,i in enumerate(file):
                if not i.startswith('NODE'):
                    loc = i.strip().split('\t')
                    loc = Location(float(loc[1]),float(loc[2]),j-1,0)
                    coords = np.append(coords,loc)
        dist_matrix = np.empty(shape=(0,len(coords)),dtype=int)
        if instance_folder=='Amazon_data':
            f = pathh+"_dist_matrix.txt"
            if path.isfile(f):
                file = open(f, "r")
                for i in file:
                    if not i.startswith('EDGE'):
                        loc = i.strip().split('\t')
                        dist_matrix = np.vstack([dist_matrix,np.array(list(map(int, loc)))])
        else:
            for i in coords:
                for j in coords:
                    dist = getdistance_euclidean(i,j)
                    dist_matrix = np.vstack([dist_matrix,dist])
                    
    else:
         raise ValueError("Failed to load the demand data: " + +instance+instance_size+data_seed  )
    
    n_parcelpoints = 0
    if instance=='Austin':
        n_parcelpoints=278
    if instance=='Seattle':
        n_parcelpoints=299
    
    adjacency = np.load(pathh+"_adjacency20.npy")#20 closest parcelpoints to each customers
    
    #service times drawn from 6-hump camelback
    service_times = calculate_service_time(coords)
    
    return coords,dist_matrix,n_parcelpoints,adjacency,service_times

def generate_demand_data(dim):
    coords = np.zeros([0])
    count = 1
    for i in range(dim):
        for j in range(dim):
            loc = Location(float(i),float(j),count,0)
            coords = np.append(coords,loc)
            count+=1
    
    return coords

def get_dist_mat_HGS(dist_matrix,loc_ids):        
    dist_mat = dist_matrix[loc_ids]
    return dist_mat[:,loc_ids]

def get_fleet(initRouteplan,num_vehicles,vehicleCapacity):
    vehicles = np.empty(shape=(0,num_vehicles))
    for v in range(num_vehicles):
        vehicles = np.append(vehicles,Vehicle(initRouteplan.copy(),vehicleCapacity,v))
    return Fleet(vehicles)

def extract_route_HGS(route,data):
    fleet = get_fleet([],data['num_vehicles'],data['vehicle_capacity'])#reset fleet and write to vehicles again
    veh = 0
    for r in route.routes:
        for i in r:
            loc = Location(data['x_coordinates'][i],data['y_coordinates'][i],data['id'][i],data['time'][i])
            idx = len(fleet["fleet"][veh]["routePlan"])-1
            fleet["fleet"][veh]["routePlan"].insert(idx,loc)
        veh+=1
    return fleet

def find_closest_parcelpoints(pathh,parcelpoints,dist_matrix,instance,data_seed):
    """
    This function is used to generate the adjacency matrix, we stored them so we do not call this function online
    """
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    shape = (len(dist_matrix)-len(parcelpoints["parcelpoints"]),len(parcelpoints["parcelpoints"]))
    adjacency = np.zeros(shape=shape,dtype=int)
    for i in range(0,len(dist_matrix)-len(parcelpoints["parcelpoints"])):
        closest = np.argsort(dist_matrix[i][-len(parcelpoints["parcelpoints"]):])[:20]#find 20 closest parcelpoints
        for j in closest:
            adjacency[i][j]=1
    pathh = pathh+sepa+'Environments'+sepa+'OOH'+sepa+'Amazon_data'+sepa+instance+sepa
    np.save(pathh+instance+"_700_"+str(data_seed)+"_adjacency20", adjacency)


def get_matrix(coords,dim,hexa=False):
    """
    For hexagon calculation, see: https://stackoverflow.com/a/7714148
    """
    max_xcoord = max(coords, key=lambda x: x.x).x
    max_ycoord = max(coords, key=lambda y: y.y).y
    min_xcoord = min(coords, key=lambda x: x.x).x
    min_ycoord = min(coords, key=lambda y: y.y).y

    customer_cells = np.empty((0,2),dtype=int)
    min_x = min_xcoord
    diff_x = max_xcoord-min_xcoord
    min_y = min_ycoord
    diff_y = max_ycoord-min_ycoord
    
    #hexa params
    if hexa:
        gridwidth = diff_x/dim
        gridheight = diff_y/dim
        c = gridwidth / 4#TODO: approximation of c, we should calculate this exactly 
        m = c / gridwidth / 2
    
    for i in coords:
        row = trunc(dim* ((i.y - min_y) / diff_y)-1e-5)     
        
        if hexa:
            rowIsOdd = row % 2 == 1   
            if rowIsOdd:#if row is odd number calculte indent of hexa grid
                column = trunc(dim* ((i.x - (gridwidth/2) - min_x) / diff_x)-1e-5)
            relative_y = i.y - (row*gridheight)
            if rowIsOdd:
                relative_x = (i.x - (column*gridwidth)) - (gridwidth/2)
            else:
                relative_x = i.x - (column*gridwidth)
            
            if relative_y < (m * relative_x) + c:#left edge
                row -=1
                if not rowIsOdd:
                    column -=1 
            elif relative_y < (-m * relative_x) - c:#rigt edge
                row -=1
                if rowIsOdd:
                    column +=1 
        else:
            column = trunc(dim* ((i.x - min_x) / diff_x)-1e-5)
            
        
        customer_cells = np.vstack((customer_cells,[column,row]))
    return customer_cells
    

class MemoryBuffer:
    """
    Pre-allocated memory interface for storing and using observations
    """
    def __init__(self, max_len, time_intervals, matrix_dim, target_dim, atype, config, stype=float32):

        self.features = torch.zeros((max_len, time_intervals, matrix_dim, matrix_dim), dtype=stype, requires_grad=False,device=config.device)
        self.target = torch.zeros((max_len, target_dim), dtype=atype, requires_grad=False,device=config.device)

        self.length = 0
        self.max_len = max_len
        self.atype = atype
        self.stype = stype
        self.config = config
        self.matrix_dim = matrix_dim
        self.time_intervals = time_intervals

    @property
    def size(self):
        return self.length

    def reset(self):
        self.length = 0

    def _get(self, idx):
        return self.features[idx], self.target[idx]

    def batch_sample(self, batch_size, randomize=True):
        if randomize:
            indices = np.random.permutation(self.length)
        else:
            indices = np.arange(self.length)

        for ids in [indices[i:i + batch_size] for i in range(0, self.length, batch_size)]:
            yield self._get(ids)

    def sample(self, batch_size):
        count = min(batch_size, self.length)
        return self._get(np.random.choice(self.length, count))

    def add(self, features, target):
        mtrx_dim = self.matrix_dim
        time_intervals = self.time_intervals
        if len(features)!=len(target):
            raise ValueError("MemoryBuffer: features and target are different length" )
        for i in range(len(features)):
            pos = self.length
            if self.length < self.max_len:
                self.length = self.length + 1
            else:
                pos = np.random.randint(self.max_len)
    
            self.features[pos] = torch.tensor(features[i].reshape(time_intervals,mtrx_dim,mtrx_dim), dtype=self.stype)
            self.target[pos] = torch.tensor(target[i][1], dtype=self.atype)
        
    def save(self, filename):
        torch.save(self.features, filename + 'feat.pt')
        torch.save(self.target, filename + 'target.pt')