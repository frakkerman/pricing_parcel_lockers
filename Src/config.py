import sys
from yaml import dump
from os import path, name
import Src.Utils.Utils as Utils
import numpy as np
import torch
from collections import OrderedDict

class Config(object):
    def __init__(self, args):

        # Path setup
        self.paths = OrderedDict()
        self.paths['root'] = path.abspath(path.join(path.dirname(__file__), '..'))

        # Reproducibility
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Copy all the variables from args to config
        self.__dict__.update(vars(args))

        # Save results after every certain number of episodes
        self.save_after = args.max_episodes // args.save_count if args.max_episodes >= args.save_count else args.max_episodes

        # Add path to models
        folder_suffix = args.experiment + args.folder_suffix
        self.paths['Experiments'] = path.join(self.paths['root'], 'Experiments')
        if args.pricing:
            self.paths['experiment'] = path.join(self.paths['Experiments'], args.env_name,'pricing', args.algo_name, folder_suffix)
        else:
            self.paths['experiment'] = path.join(self.paths['Experiments'], args.env_name,'offering', args.algo_name, folder_suffix)
            
        if name == 'nt':
             suffix = '\\'
        else:
             suffix = '/'
        path_prefix = [self.paths['experiment'], str(args.seed)]
        self.paths['logs'] = path.join(*path_prefix, 'Logs'+suffix)
        self.paths['checkpoint'] = path.join(*path_prefix, 'Checkpoints'+suffix)
        self.paths['results'] = path.join(*path_prefix, 'Results'+suffix)

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'datasets', 'data']:
                Utils.create_directory_tree(val)

        # Save the all the configuration settings
        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)

        # Output logging
        sys.stdout = Utils.Logger(self.paths['logs'], args.log_output)

        #load data
        if args.load_data:
            self.coords,self.dist_matrix,self.n_parcelpoints,self.adjacency,self.service_times = Utils.load_demand_data(self.paths['root'],args.instance,args.data_seed,args.clip_service_time,args.truck_speed)
            self.coords_test,self.dist_matrix_test,self.n_parcelpoints_test,self.adjacency_test,self.service_times_test = Utils.load_demand_data(self.paths['root'],args.instance,args.data_seed_test,args.clip_service_time,args.truck_speed)
        else:#only used for debug purpose
            self.coords,self.dist_matrix,self.n_parcelpoints,self.adjacency = Utils.generate_demand_data(100),[],6,np.ones(6),np.ones(100)
            self.coords_test,self.dist_matrix_test,self.n_parcelpoints_test,self.adjacency_test,self.service_times_test = Utils.generate_demand_data(100),[],6,np.ones(6),np.ones(100)

        # Get the domain and algorithm
        self.env = self.get_domain(args.env_name, args=args,path=path.join(self.paths['root'], 'Environments'))
        self.env.seed(seed)
        
        self.test_env = self.get_domain(args.env_name, args=args,path=path.join(self.paths['root'], 'Environments'),test_env=True)
        self.test_env.seed(seed)

        # Set Model
        self.algo = Utils.dynamic_load(path.join(self.paths['root'], 'Src', 'Algorithms'), args.algo_name, load_class=True)

        # GPU
        if args.gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.cuda = 0
        if self.device.type=="cuda":
            print('Number of GPUs available: ',torch.cuda.device_count())
            self.cuda = 1

        # optimizer
        if args.optim == 'adam':
            self.optim = torch.optim.Adam
        elif args.optim == 'rmsprop':
            self.optim = torch.optim.RMSprop
        elif args.optim == 'sgd':
            self.optim = torch.optim.SGD
        else:
            raise ValueError('Undefined type of optmizer')


        print("=====Configurations=====\n", args)

    # Load the domain
    def get_domain(self, tag, args, path,test_env=False):
        if tag[:11] == 'Parcelpoint':
            obj = Utils.dynamic_load(path, tag, load_class=True)
            if test_env:
                env = obj(model=args.algo_name,max_steps_r=args.max_steps_r,max_steps_p=args.max_steps_p,pricing=args.pricing,n_vehicles=args.n_vehicles,
                      veh_capacity=args.veh_capacity,parcelpoint_capacity=args.parcelpoint_capacity,fraction_capacitated=args.fraction_capacitated,incentive_sens=args.incentive_sens,base_util=args.base_util,
                      home_util=args.home_util,reopt=args.reopt,load_data=args.load_data,coords=self.coords_test,dist_matrix=self.dist_matrix_test,
                      n_parcelpoints=self.n_parcelpoints_test,adjacency=self.adjacency_test,service_times=self.service_times_test,dissatisfaction=self.dissatisfaction,hgs_time=args.hgs_reopt_time)
            else:
                env = obj(model=args.algo_name,max_steps_r=args.max_steps_r,max_steps_p=args.max_steps_p,pricing=args.pricing,n_vehicles=args.n_vehicles,
                      veh_capacity=args.veh_capacity,parcelpoint_capacity=args.parcelpoint_capacity,fraction_capacitated=args.fraction_capacitated,incentive_sens=args.incentive_sens,base_util=args.base_util,
                      home_util=args.home_util,reopt=args.reopt,load_data=args.load_data,coords=self.coords,dist_matrix=self.dist_matrix,
                      n_parcelpoints=self.n_parcelpoints,adjacency=self.adjacency,service_times=self.service_times,dissatisfaction=self.dissatisfaction,hgs_time=args.hgs_reopt_time)
            return env

if __name__ == '__main__':
    pass