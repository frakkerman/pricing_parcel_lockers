# Pricing and Offering of Out-of-Home Delivery

This project contains code for the paper titled "Pricing and Offering of Out-of-home Delivery", see: 


## Environment

The code is written in Python 3.10. We use PyTorch 1.13.1 to model neural network architectures. A requirements.txt details further requirements of our project. We tested our project on a Windows 11 environment, and a high-performance cluster, whose details you find here:


## Folder Structure
The repository contains the following folders:

Src<br>		|-------Algorithms<br>     	  |-------Utils<br>
Environments <br>	|-------OOH


On the first level you can see run.py which implements the overall policy training and evaluation loop.

### Src 

On the first level you can see a parser.py, wherein we set hyperparameters and environment variables, and config.py, which preprocesses inputs.


`Algorithms`: 
* Heuristic.py: Conntains the benchmark heuristic by Yang et al. (2016), and the "Offer_all" baseline heuristic.
* Agent.py: Groups several high-level agent functionalities

`Utils`: 
* Actor.py and Critic.py: Contain the neural network architectures for actor and critic respectively.
* Basis.py: Contains the state representation module.
* Utils.py: Contains several helper functions such as plotting

### Environments
* `OOH`: Contains the implementation of the OOH environments and the used data.


## To the make the code work

 * Create a local python environment by subsequently executing the following commands in the root folder
	* `python3 -m venv venv`
	* `source venv/bin/activate`
	* `python -m pip install -r requirements.txt`
	* `deactivate`

 * `Src/parser.py` Set your study's hyperparameters in this file, e.g., which environment to use or setting learning rates
 
 * `run.py` Execute this file using the command line `python3 run.py`.
 
 * Note that you might have to adapt your root folder's name to `ooh_code`
 
## License
* [MIT license](https://opensource.org/license/mit/)
* Copyright 2023 © [Fabian Akkerman](https://people.utwente.nl/f.r.akkerman), [Peter Dieter](https://en.wiwi.uni-paderborn.de/dep3/schryen/team/dieter/88592), [Martijn Mes](https://www.utwente.nl/en/bms/iebis/staff/mes/), [Guido Schyren](https://en.wiwi.uni-paderborn.de/dep3/schryen/team/schryen/72850)

## Bibliography

Choice-Based Demand Management and Vehicle Routing in E-Fulfillment
Xinan Yang, Arne K. Strauss, Christine S. M. Currie, and Richard Eglese
Transportation Science 2016 50:2, 473-488 