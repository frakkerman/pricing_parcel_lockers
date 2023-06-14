from __future__ import print_function
import numpy as np
import torch
from torch import float32
import torch.nn as nn
import shutil
import matplotlib.pyplot as plt
from os import path, mkdir, listdir, fsync, name
import importlib
from time import time
import sys
from Environments.OOH.containers import Location,Vehicle,Fleet
from math import trunc

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
    np.save(config.paths['results'] + "test_returns_mean", test_returns)
    np.save(config.paths['results'] + "test_returns_std", test_std)
    np.save(config.paths['results'] + "step_time", step_time)
    x = config.save_after * np.arange(0, len(test_returns))
    plt.figure()
    plt.ylabel("Total return")
    plt.xlabel("Episode")
    plt.title("Performance")
    plt.plot(x,test_returns,color='#CC4F1B')
    plt.fill_between(x,np.array(test_returns)+np.array(test_std),np.array(test_returns)-np.array(test_std),alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.savefig(config.paths['results'] + "performance_test_runs.png")
    plt.close()

def save_plots_stats(stats,costs,run_time,actions,config,episode):
    info = np.array(stats)
    np.save(config.paths['results'] + "time", run_time)
    
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
    plt.savefig(config.paths['results'] + "bar_deliveries.png")
    plt.close()
    
    #4 boxplot final hgs distance
    plt.figure()
    plt.ylabel("Total distance")
    plt.title("Performance")
    plt.boxplot(costs)
    plt.savefig(config.paths['results'] + "box_hgs_distance.png")
    plt.close()
    
    #save data
    np.save(config.paths['results'] + "hgs_costs", costs)
    np.save(config.paths['results'] + "num_home_deliveries", count_home)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.ctr = 0
        self.nan_check_fequency = 10000

    # def custom_weight_init(self):
    #     # Initialize the weight values
    #     for m in self.modules():
    #         weight_init(m)

    def update(self, loss, retain_graph=False, clip_norm=False):
        self.optim.zero_grad()  # Reset the gradients
        loss.backward(retain_graph=retain_graph)
        self.step(clip_norm)

    def step(self, clip_norm):
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        self.optim.step()
        self.check_nan()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def check_nan(self):
        # Check for nan periodically
        self.ctr += 1
        if self.ctr == self.nan_check_fequency:
            self.ctr = 0
            # Note: nan != nan  #https://github.com/pytorch/pytorch/issues/4767
            for nme, param in self.named_parameters():
                if (param != param).any():
                    raise ValueError(nme + ": Weights have become nan... Exiting.")

    def reset(self):
        return


def binaryEncoding(num, size,level=1):
    binary = np.zeros(size)
    i = -1
    while num > 0:
        binary[i] = num % (level+1)
        num = num//(level+1)
        i -= 1
    return binary


def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
    Computationally more expensive? Maybe, Not sure.
    adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    y_t = torch.transpose(y, 0, 1)

    # a^2 + b^2 - 2ab
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # dist[dist != dist] = 0 # replace nan values with 0
    return dist


class Space:
    def __init__(self, low=[0], high=[1], dtype=np.uint8, size=-1):
        if size == -1:
            self.shape = np.shape(low)
        else:
            self.shape = (size, )
        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype
        self.n = len(self.low)


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


def clip_norm(params, max_norm=1):
    # return params
    norm_param = []
    for param in params:
        norm = np.linalg.norm(param, 2)
        if norm > max_norm:
            norm_param.append(param/norm * max_norm)
        else:
            norm_param.append(param)
    return norm_param

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
                        loc = Location(int(loc[0]),int(loc[1]),0,0)#add id_num here
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

def load_demand_data(pathh,city,data_seed):
    if name == 'nt':#windows
        sepa= '\\'
    else:
        sepa='/'
    pathh = pathh+sepa+'Environments'+sepa+'OOH'+sepa+'Amazon_data'+sepa+city+sepa
    if path.exists(pathh):
        pathh = pathh+city+'_700_'+str(data_seed)
        f = pathh+"_coords.txt"
        if path.isfile(f):
            file = open(f, "r")
            coords = np.zeros([0])
            for j,i in enumerate(file):
                if not i.startswith('NODE'):
                    loc = i.strip().split('\t')
                    loc = Location(float(loc[1]),float(loc[2]),j-1,0)
                    coords = np.append(coords,loc)
        f = pathh+"_dist_matrix.txt"
        dist_matrix = np.empty(shape=(0,len(coords)),dtype=int)
        if path.isfile(f):
            file = open(f, "r")
            for i in file:
                if not i.startswith('EDGE'):
                    loc = i.strip().split('\t')
                    dist_matrix = np.vstack([dist_matrix,np.array(list(map(int, loc)))])
    else:
         raise ValueError("Failed to load the demand data: " + +city+'_700_'+data_seed  )
    
    n_parcelpoints = 0
    if city=='Austin':
        n_parcelpoints=278
    if city=='Seattle':
        n_parcelpoints=299
    
    adjacency = np.load(pathh+"_adjacency20.npy")#20 closest parcelpoints to each customers
    
    return coords,dist_matrix,n_parcelpoints,adjacency

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

def find_closest_parcelpoints(pathh,parcelpoints,dist_matrix,city,data_seed):
    """
    This function is used to generate the adjacency matrix, we stored them so we do not call this function
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
    pathh = pathh+sepa+'Environments'+sepa+'OOH'+sepa+'Amazon_data'+sepa+city+sepa
    np.save(pathh+city+"_700_"+str(data_seed)+"_adjacency20", adjacency)


def get_matrix(coords,dim):
    max_xcoord = max(coords, key=lambda x: x.x).x
    max_ycoord = max(coords, key=lambda y: y.y).y
    min_xcoord = min(coords, key=lambda x: x.x).x
    min_ycoord = min(coords, key=lambda y: y.y).y
    
    customer_cells = np.empty((0,2),dtype=int)
    min_x = min_xcoord
    diff_x = max_xcoord-min_xcoord
    min_y = min_ycoord
    diff_y = max_ycoord-min_ycoord
    
    for i in coords:
        column = trunc(dim* ((i.x - min_x) / diff_x)-1e-5)
        row = trunc(dim* ((i.y - min_y) / diff_y)-1e-5)
        customer_cells = np.vstack((customer_cells,[column,row]))
    return customer_cells

class MemoryBuffer:
    """
    Pre-allocated memory interface for storing and using observations
    """
    def __init__(self, max_len, matrix_dim, target_dim, atype, config, stype=float32):

        self.features = torch.zeros((max_len, matrix_dim, matrix_dim), dtype=stype, requires_grad=False)
        self.target = torch.zeros((max_len, target_dim), dtype=atype, requires_grad=False)

        self.length = 0
        self.max_len = max_len
        self.atype = atype
        self.stype = stype
        self.config = config
        self.matrix_dim = matrix_dim

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
        if len(features)!=len(target):
            raise ValueError("MemoryBuffer: features and target are different length" )
        for i in range(len(features)):
            pos = self.length
            if self.length < self.max_len:
                self.length = self.length + 1
            else:
                pos = np.random.randint(self.max_len)
    
            self.features[pos] = torch.tensor(features[i].reshape(mtrx_dim,mtrx_dim), dtype=self.stype)
            self.target[pos] = torch.tensor(target[i][1], dtype=self.atype)
        
    def save(self, filename):
        torch.save(self.features, filename + 'feat.pt')
        torch.save(self.target, filename + 'target.pt')