This is the code base for CS 558 Project by Falak Mandali and Samitha Ranasinghe. 

## Python environment

Setup a conda environment,

`conda create --name cs558_project python=3.11`

activate,

`conda activate cs558_project`

and install the required packages,

`pip install -r requirements.txt`

## Generating expert dataset
For generating dataset, run,

`python gen_dataset.py`

For visualizing a example RRT* run,

`python visualize_rrt.py`

## Neural network

For neural network architecture.

`python nn.py`

For training,

`python train_nn.py`

For testing,

`python test_nn.py`

To produce performance evaluation graphs and metrics for the model,

`python test_nn_graphs.py`

## MPC
To simulate and navigate robot specify terminal cost type in `config.yaml` and run,

`python mpc.py`

To reproduce path plots,

`python plot_mpc_path.py`

