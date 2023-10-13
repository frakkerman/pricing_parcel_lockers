import argparse
from datetime import datetime


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Seed for reproducibility
        parser.add_argument("--seed", default=1234, help="seed for variance testing",type=int)

        # General parameters
        parser.add_argument("--save_count", default=50, help="Number of checkpoints for saving results and model", type=int)
        parser.add_argument("--log_output", default='term_file', help="Log all the print outputs",choices=['term_file', 'term', 'file'])
        parser.add_argument("--debug", default=True, type=self.str2bool, help="Debug mode on/off")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="flag to save model checkpoints")

        # For documentation purposes
        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='default', help="folder name suffix")
        parser.add_argument("--experiment", default='run', help="Name of the experiment")
        
        parser.add_argument("--algo_name", default='PPO', help="Policy/algorithm used, capital sensitive",choices=['ML_Foresight','Heuristic','Baseline','PPO'])
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)
        
        # Environment parameters
        self.environment_parameters(parser)  
        
        # General settings for algorithms
        self.ML_parameters(parser)
        self.Heuristic_parameters(parser)
        self.Baseline_parameters(parser)
        self.PPO_parameters(parser)
        
        self.parser = parser

    def environment_parameters(self, parser):
        parser.add_argument("--env_name", default='Parcelpoint_py', help="Environment to run the code")
        parser.add_argument("--max_episodes", default=int(2500), help="maximum number of training episodes", type=int)
        
        parser.add_argument("--max_steps_r", default=90, help="maximum customers per episode r of gamma dist.", type=int)#700
        parser.add_argument("--max_steps_p", default=0.5, help="maximum customers per episode p of gamma dist. [0,1]", type=float)
        
        parser.add_argument("--load_data", default=True, help="whether to load location data from file or to generate data (only used for debug)", type=self.str2bool)
        parser.add_argument("--instance", default='RC', help="which instance to load",choices=['Austin','Seattle','C','R','RC'])
        parser.add_argument("--data_seed", default=0, help="which dataset seed to load for training",choices=[0,1,2,3], type=int)#0-3 for Amazon, 0-1 for Homberger
        parser.add_argument("--data_seed_test", default=1, help="which dataset seed to load for testing",choices=[0,1,2,3], type=int)#0-3 for Amazon, 0-1 for Homberger
        
        parser.add_argument("--pricing", default=True, help="if we use pricing or offering decision space", type=self.str2bool)
        parser.add_argument("--max_price", default=2.0, help="max delivery charge >0", type=float)
        parser.add_argument("--min_price", default=-10.0, help="max discount <0", type=float)
        
        parser.add_argument("--k", default=19, help="Number of parcelpoints to offer to customer", type=int)
        
        parser.add_argument("--n_vehicles", default=10, help="number of vehicles", type=int)#Austin=20, Seattle=25
        parser.add_argument("--veh_capacity", default=9, help="capacity per vehicle per day", type=int)
        parser.add_argument("--parcelpoint_capacity", default=100000, help="parcel point capacity per day", type=int)
        
        parser.add_argument("--incentive_sens", default=-0.175, help="sensitivty of customer to incentives", type=float)#-0.25
        parser.add_argument("--base_util", default=-2.0, help="base utility across all alternativesy", type=float)#-2.0
        parser.add_argument("--home_util", default=3.2, help="utility given to home delivery", type=float)#3.55 amazon, 3.1 homberger
        parser.add_argument("--dissatisfaction", default=False, help="customer dissatisfaction penalty when all delivery options have too high prices", type=self.str2bool)
        
        parser.add_argument("--revenue", default=90, help="revenue per customer", type=float)#90
        parser.add_argument("--fuel_cost", default=0.6, help="costs of fuel per distance unit", type=float)#0.3/0.6
        parser.add_argument("--truck_speed", default=30, help="distance travelled per hour", type=float)#30
        parser.add_argument("--clip_service_time", default=10, help="maximum service time in minutes", type=float)#10
        parser.add_argument("--driver_wage", default=30, help="salary of driver per hour", type=float)#30
        
        parser.add_argument("--home_failure", default=0.1, help="the probability of delivery failure for home delivery", type=float)#0.1
        parser.add_argument("--failure_cost", default=10.0, help="the monetary costs of a delivery failure", type=float)#10
        
        parser.add_argument("--reopt", default=100000, help="re-opt frequency of cheapest insertion route using HGS", type=int)
        parser.add_argument("--hgs_reopt_time", default=1.1, help="re-opt HGS time limit", type=float)
        
        parser.add_argument("--hgs_final_time", default=1.1, help="HGS time limit for obtaining final routing schedule", type=float)
        
    def ML_parameters(self, parser):
        parser.add_argument("--grid_dim", default=10, help="division of operational area in X*X clusters", type=int)
        parser.add_argument("--hexa", default=False, help="division of operational area in hexagional grid instead of squares (beta)", type=self.str2bool)
        parser.add_argument("--n_input_layers", default=3, help="divide feature map in X time intervals", type=int)
        parser.add_argument("--only_phase_one", default=False, help="when True, we stop learning after an initial data collection phase", type=self.str2bool)
        parser.add_argument("--initial_phase_epochs", default=30, help="maximum number of training epochs", type=int)
        parser.add_argument("--buffer_size", default=int(1000), help="Size of memory buffer", type=int)
        parser.add_argument("--batch_size", default=8, help="Batch size", type=int)
        parser.add_argument("--learning_rate", default=1e-3, help="learning rate", type=float)
        
        parser.add_argument("--init_theta_cnn", default=1.0, help="initial weight for cheapest insertion in historic route, [0,1]", type=float)
        parser.add_argument("--cool_theta_cnn", default=(1/100), help="weight reduction for cheapest insertion", type=float)
        
        #parser.add_argument("--load_embed", default=False, type=self.str2bool, help="Retrain flag, if True we do not retrain but try to load a stored model")
        parser.add_argument("--optim", default='adam', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--use3d_conv", default=False, type=self.str2bool, help="Use 3D convolution instead of 2D")
        parser.add_argument("--n_filters", default=16, help="number of filters in first convolutional layer (2nd is 2*X)", type=int)
        parser.add_argument("--dropout", default=0.1, help="dropout rate of the FC layers", type=float)
       
    def Heuristic_parameters(self, parser):
        parser.add_argument("--init_theta", default=1.0, help="weight for cheapest insertion in historic route, [0,1]", type=float)
        parser.add_argument("--cool_theta", default=1/90, help="weight reduction for cheapest insertion", type=float)
    
    def Baseline_parameters(self, parser):
        parser.add_argument("--save_routes", default=False, help="Used to generate and save routes for use inside Heuristic", type=self.str2bool)#could consider to make an updating loop for routes
        parser.add_argument("--price_pp", default=-0.0, help="fixed fee price to offer for all parcelpoints", type=float)
        parser.add_argument("--price_home", default=0.0, help="fixed fee price to offer for home delivery", type=float)
        
    def PPO_parameters(self,parser):
        parser.add_argument("--actor_lr", default=1e-4, help="(1e-2) Learning rate of actor", type=float)
        parser.add_argument("--critic_lr", default=1e-2, help="(1e-2) Learning rate of critic", type=float)
        parser.add_argument("--state_lr", default=1e-1, help="Learning rate of state features", type=float)
        parser.add_argument("--batch_size_ppo", default=100, help="Batch size", type=int)
        parser.add_argument("--fourier_coupled", default=True, help="Coupled or uncoupled fourier basis", type=self.str2bool)
        parser.add_argument("--fourier_order", default=3, help="Order of fourier basis, " + "(if > 0, it overrides neural nets)", type=int)
        parser.add_argument("--hiddenLayerSize", default=16, help="size of hiddenlayer of critic", type=int)
        parser.add_argument("--hiddenActorLayerSize", default=8, help="size of hiddenlayer", type=int)
        parser.add_argument("--gamma", default=0.999, help="Discounting factor", type=float)
        parser.add_argument("--gauss_variance", default=2, help="Variance for gaussian policy", type=float) # 1 original setting
        parser.add_argument("--clipping_factor",default=0.2, help = "PPO clipping factor",type = float)
        parser.add_argument("--td_lambda", default=0.95, help="lambda factor for calculating advantages", type=float)
        parser.add_argument("--policy_update_epochs", default=25,help="number of epochs with which we perform policy updates in PPO", type=int)
        parser.add_argument("--critic_update_epochs", default=25,help="number of epochs with which we perform critic updates in PPO", type=int)
        
        
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

    