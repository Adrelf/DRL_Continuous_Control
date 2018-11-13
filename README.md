# DRL_Continuous_Control
 Train a double-jointed arm to reach target locations

# DRL-navigation
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
Step 1: Install ML-agents ==> https://github.com/Unity-Technologies/ml-agents and follow the instructions here ==> https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md.

Step 2: Install Python (only version >3 is supported) and PyTorch.

Step 3: Clone this repository.

Step 4: Download the Unity Environment ==> https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
Then, place the file in the DRL_Continuous_Control/ folder in this repository, and unzip (or decompress) the file.

To train an agent, please use the following command:
$python main_DDPG.py
