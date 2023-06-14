import argparse
from datetime import datetime


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Seed for reproducibility
        parser.add_argument("--seed", default=1234, help="seed for variance testing",type=int)

        # General parameters
        parser.add_argument("--save_count", default=1000, help="Number of checkpoints for saving results and model", type=int)
        parser.add_argument("--optim", default='sgd', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--log_output", default='term_file', help="Log all the print outputs",choices=['term_file', 'term', 'file'])
        parser.add_argument("--debug", default=True, type=self.str2bool, help="Debug mode on/off")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="flag to save model checkpoints")

        # For documentation purposes
        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='default', help="folder name suffix")
        parser.add_argument("--experiment", default='run', help="Name of the experiment")
        
        # Environment parameters
        self.environment_parameters(parser)  
        
        # General settings for algorithms
        self.ML_parameters(parser)
        self.Heuristic_parameters(parser)
        
        self.parser = parser

    def environment_parameters(self, parser):
        parser.add_argument("--algo_name", default='ML_Foresight', help="RL algorithm",choices=['ML_Foresight','Heuristic'])
        parser.add_argument("--env_name", default='Parcelpoint_py', help="Environment to run the code")
        parser.add_argument("--max_episodes", default=int(10000), help="maximum number of episodes", type=int)
        
        parser.add_argument("--max_steps_r", default=700, help="maximum steps per episode r of gamma dist.", type=int)
        parser.add_argument("--max_steps_p", default=0.5, help="maximum steps per episode p of gamma dist. [0,1]", type=float)
        
        parser.add_argument("--load_data", default=True, help="whether to load location data from file or to generate data", type=self.str2bool)
        parser.add_argument("--city", default='Austin', help="which city to load",choices=['Austin','Seattle'])
        parser.add_argument("--data_seed", default=0, help="which dataset to load",choices=[0,1,2,3], type=int)
        
        parser.add_argument("--pricing", default=True, help="if we use pricing or offering decision space", type=self.str2bool)
        parser.add_argument("--max_price", default=5.0, help="max delivery charge >0", type=float)
        parser.add_argument("--min_price", default=-5.0, help="max discount <0", type=float)
        
        parser.add_argument("--k", default=1, help="Number of parcelpoints to offer to customer", type=int)
        
        parser.add_argument("--n_vehicles", default=2, help="number of vehicles", type=int)
        parser.add_argument("--veh_capacity", default=60, help="capacity per vehicle per day", type=int)
        parser.add_argument("--parcelpoint_capacity", default=10000, help="parcel point capacity per day", type=int)
        
        parser.add_argument("--incentive_sens", default=-0.99, help="sensitivty of customer to incentives <0", type=float)
        parser.add_argument("--base_util", default=0.2, help="base utility across all alternativesy", type=float)
        parser.add_argument("--home_util", default=0.3, help="utility given to home delivery", type=float)
        
        parser.add_argument("--revenue", default=0.4, help="revenue per customer", type=float)
        parser.add_argument("--fuel_cost", default=0.1, help="costs of fuel per distance unit", type=float)
        parser.add_argument("--truck_speed", default=40, help="distance travelled per hour", type=float)
        parser.add_argument("--del_time", default=5, help="time in minutes to drop off parcel", type=float)
        parser.add_argument("--driver_wage", default=25, help="salary of driver per hour", type=float)
        
        parser.add_argument("--reopt", default=10000000, help="re-opt frequency of cheapest insertion route using HGS", type=int)
        
    def ML_parameters(self, parser):
        parser.add_argument("--grid_dim", default=10, help="division of operational area in X*X clusters", type=int)
        parser.add_argument("--n_input_layers", default=3, help="divide feature map in X time intervals", type=int)
        parser.add_argument("--only_phase_one", default=False, help="when True, we stop learning after an initial data collection phase", type=self.str2bool)
        parser.add_argument("--initial_phase_epochs", default=1000, help="maximum number of episodes", type=int)
        parser.add_argument("--buffer_size", default=int(30), help="Size of memory buffer", type=int)
        parser.add_argument("--batch_size", default=1, help="Batch size", type=int)
        #lr, epochs, layers+size
        parser.add_argument("--init_theta_cnn", default=1.0, help="weight for cheapest insertion in historic route, [0,1]", type=float)
        parser.add_argument("--cool_theta_cnn", default=True, help="weight reduction for cheapest insertion", type=self.str2bool)
        
        parser.add_argument("--load_embed", default=False, type=self.str2bool, help="Retrain flag, if True we do not retrain but try to load a stored model")
        parser.add_argument("--use3d_conv", default=False, type=self.str2bool, help="Use 3D convolution instead of 2D")
        
    def Heuristic_parameters(self, parser):
        parser.add_argument("--save_routes", default=False, help="Used to save routes for use inside heuristic", type=self.str2bool)#could consider to make this an updating loop
        parser.add_argument("--init_theta", default=1.0, help="weight for cheapest insertion in historic route, [0,1]", type=float)
        parser.add_argument("--cool_theta", default=(1/1500), help="weight reduction for cheapest insertion", type=float)
        
        parser.add_argument("--offer_all", default=False, help="baseline heuristic, just offer all feasible", type=self.str2bool)
        
    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser

    