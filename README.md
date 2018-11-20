# DRL_Continuous_Control
Train a double-jointed arm that can move to target locations. The goal of this agent is to maintain its position at the target location for as many time steps as possible.
The deep reinforcement learning algorithm is based on actor-critic method (DDPG).

![alt text](https://github.com/Adrelf/DRL_Continuous_Control/blob/master/images/reacher.gif)
        
      
# The Environment 
The environment is determinist.
 + State: 
 The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

 + Actions:
 Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

 + Reward strategy:
    - +0.1 is provided for each step that the agent's hand is in the goal location.

 + Solved Requirements:
    - The agent must get an average score of +30 over 100 consecutive episodes.

# Algorithm
DDPG with hyperparameters tuning.

PPO ==> WIP.
 
# Getting started

## Dependencies
 * Python 3.6 or higher
 * PyTorch
 * Create (and activate) a new environment with Python 3.6:
        ```
        conda create --name drlnd python=3.6<br/>
        source activate drlnd
        ```
 * Install requirements:
        ```
        clone git https://github.com/Adrelf/DRL_Continuous_Control.git <br/>
        cd DRL_Continuous_Control<br/>
        pip install -e .<br/>
        ```
 * Download the [Unity Environment!] (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)   
Then, place the file in the DRL_Continuous_Control/ folder in this repository, and unzip (or decompress) the file.

# Instructions
 * To train an agent, please use the following command:
        ```
        $python main_DDPG.py
        ```
 with the fowolling hyper-parameters:
 
 Parameters | Value | Description
----------- | ----- | -----------
BUFFER_SIZE | int(1e6) | replay buffer size
BATCH_SIZE | 1024 | minibatch size
GAMMA | 0.99 | discount factor
TAU | 1e-3 | for soft update of target parameters
LR_ACTOR | 1e-4 | learning rate of the actor
LR_CRITIC | 1e-3 | learning rate of the critic
WEIGHT_DECAY | 0 | L2 weight decay
NUM_AGENTS | 20 | Number of agents
fc1_units | 128 | Number of nodes in first hidden layer for actor
fc2_units | 56 | Number of nodes in second hidden layer for actor
fc1_units |256 | Number of nodes in first hidden layer for critic
fc2_units | 128 | Number of nodes in second hidden layer for critic
