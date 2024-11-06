# Dynamic Selection and Pricing of Parcel Locker Delivery

This project contains code for the paper titled "Learning Dynamic Selection and Pricing of Out-of-Home Deliveries" by Fabian Akkerman, Peter Dieter, and Martijn Mes, see: [this paper](https://doi.org/10.1287/trsc.2023.0434)

A preprint version of the paper can be found on ArXiv, see: [this link](https://arxiv.org/abs/2311.13983)

## Citation

When using the code or data in this repo, please cite the following work:

```
@article{akkerman2024learning,
      title={Learning Dynamic Selection and Pricing of Out-of-Home Deliveries}, 
      author={Fabian Akkerman and Peter Dieter and Martijn Mes},
      year={2024},
      journal = {Transportation Science}
}
```

When using the Amazon or Gehring and Homberger instances, ensure to also cite the respective works from which we abstracted the data, see the citation at the bottom of this README.

## Environment

The code is written in Python 3.10. We use PyTorch 2.0.0 to model neural network architectures and hygese 0.0.0.8, the Python HGS implementation. A requirements.txt details further requirements of our project. We tested our project on a Windows 11 environment, and a high-performance Linux cluster.


## Folder Structure
The repository contains the following folders:

- **`Environments/`**: Contains the problem and all related data.
- **`Src/`**: Contains the main code for all algorithms.
  - **`Algorithms/`**: Contains all algorithmic implementations.
  - **`Utils/`**: Contains utility functions, e.g., data loading, actor and critic class structure, prediction models.


On the first level you can see run.py which implements the overall policy training and evaluation loop. For running PPO we use a seperate file called run_ppo.py.

### Src 

On the first level you can see a parser.py, wherein we set hyperparameters and environment variables, and config.py, which preprocesses inputs.


`Algorithms`: 
* Agent.py: Groups several high-level agent functionalities
* Baseline.py: Contains the baseline, StaticPricing.
* DSPO.py: Contains our proposed contribution, Dynamic Selection and Pricing of OOH (DSPO)
* Heuristic.py: Conntains the benchmark heuristics by Yang et al. (2016).
* PPO.py: Contains the Gaussian PPO policy, as proposed in Schulman et al. (2017)

`Utils`: 
* Actor.py and Critic.py: Contain the neural network architectures for actor and critic respectively.
* Basis.py: Contains the state representation module.
* Predictors.py: Contains the prediction models used for DSPO and the linear benchmark.
* Utils.py: Contains several helper functions such as plotting.

### Environments
`OOH` Contains the implementation of the OOH environment and the used data (Amazon_data and HombergerGehring_data).
* containers.py: container @dataclasses for storing during simulation.
* customerchoice.py: the MNL choice model.
* env_utils.py: some utility functions related to the environment.
* Parcelpoint_py.py:the main problem implementation, following Gymnasium implementation structure mainly.


## To the make the code work

 * Create a local python environment by subsequently executing the following commands in the root folder
	* `python3 -m venv venv`
	* `source venv/bin/activate`
	* `python -m pip install -r requirements.txt`
	* `deactivate`

 * `Src/parser.py` Set your study's hyperparameters in this file.
 
 * `run.py` Execute this file using the command line `python3 run.py`. Run the PPO algorithm with `python3 run_ppo.py`
 
 * Note that you might have to adapt your root folder's name to `ooh_code`
 
 * Note that `hygese.py` requires a slight change to one source file when running with `--load_data=True`, this change is indicated when running the code
 
## Contributing

If you have proposed extensions to this codebase, feel free to do a pull request! If you experience issues, feel free to send us an email.

## License
* [MIT license](https://opensource.org/license/mit/)
* Copyright 2024 © [Fabian Akkerman](https://people.utwente.nl/f.r.akkerman), [Peter Dieter](https://en.wiwi.uni-paderborn.de/dep3/schryen/team/dieter), [Martijn Mes](https://www.utwente.nl/en/bms/iebis/staff/mes/)

## Bibliography

### Data sets used

Gehring, H., Homberger, J. (2002). Parallelization of a Two-Phase Metaheuristic for Routing Problems with Time Windows. Journal of Heuristics 8, 251–276.

Daniel Merchan, Jatin Arora, Julian Pachon, Karthik Konduri, Matthias Winkenbach, Steven Parks, & Joseph Noszek (2022). 2021 Amazon last mile routing research challenge: Data set. Transportation Science.  

### Benchmarks

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017). Proximal Policy Optimization Algorithms.

Yang, X., Strauss, A., Currie, C., & Eglese, R. (2016). Choice-Based Demand Management and Vehicle Routing in E-Fulfillment. Transportation Science, 50(2), 473-488.

