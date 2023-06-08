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
        self.save_after = args.max_episodes // args.save_count if args.max_episodes > args.save_count else args.max_episodes

        # Add path to models
        folder_suffix = args.experiment + args.folder_suffix
        self.paths['Experiments'] = path.join(self.paths['root'], 'Experiments')
        self.paths['experiment'] = path.join(self.paths['Experiments'], args.env_name, args.algo_name, folder_suffix)

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

        # Get the domain and algorithm
        self.env, self.gym_env, self.cont_actions = self.get_domain(args.env_name, args=args, debug=args.debug,
                                                               path=path.join(self.paths['root'], 'Environments'))
        self.env.seed(seed)


        # Hiddenlayer size
        self.hiddenLayerSize = args.hiddenLayerSize

        # Set Model
        self.algo = Utils.dynamic_load(path.join(self.paths['root'], 'Src', 'Algorithms'), args.algo_name, load_class=True)

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
    def get_domain(self, tag, args, path, debug=True):
        if tag[:11] == 'Parcelpoint':
            obj = Utils.dynamic_load(path, tag, load_class=True)
            env = obj(model=args.algo_name,pricing=args.pricing,n_vehicles=args.n_vehicles,veh_capacity=args.veh_capacity,parcelpoint_capacity=args.parcelpoint_capacity,
                      incentive_sens=args.incentive_sens,base_util=args.base_util,home_util=args.home_util,reopt=args.reopt,
                      saveRoutes=args.saveRoutes,path=self.paths['root'],load_data=args.load_data,city=args.city,data_seed=args.data_seed,num_offers=args.k)
            return env, False, env.action_space.dtype == np.float32

if __name__ == '__main__':
    pass