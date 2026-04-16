Setup a conda environment,

`conda create --name cs558_project python=3.11`

activate,

`conda activate cs558_project`

and install the required packages,

`pip install -r requirements.txt`

For generating dataset, run,

`python gen_dataset.py`

For visualizing a example RRT* run,

`python visualize_rrt.py`

For training the neural network,

`python train_nn.py`

To produce performance evaluation graphs and metrics for the model,

`python test_nn_graphs.py`

To produce mpc results, specify terminal cost type in `config.yaml` and run.

`mpc.py`