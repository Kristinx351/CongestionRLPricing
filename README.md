# Folder description
## generate_roadnet folder

A Python pipeline using NetworkX to convert real-world data (Manhattan, Porto and Hangzhou) from OpenStreetMap and taxi trajectories into a simulator-compatible format, including traffic flow, signals, intersections and roads data. 

## run folder

`run_XX.py` :  contains training process of different baselines (partial).

## actor_critic folder

Contains realizations of several deep learning and reinforcement learning algorithms (GCN, Actor-Critic network)

## agent folder

Contains several agents with different behavior modes in the Multi-Agent RL environment (eg. roads, drivers).

## metric folder

Contains some API calculating metrics for agents’ cost.

## frontend folder

Contains some fronted tools for traffic visualization.

You could use `index.html` and the generated `.txt` file. 
Run 
> python download_replay.py

to download example replay txt files after you finish the training process.
Checkout [Document](https://cityflow.readthedocs.io/en/latest/replay.html) for more instructions.
