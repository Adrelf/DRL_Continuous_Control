from unityagents import UnityEnvironment
import gym
import time
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parallelEnv import parallelEnv
import PPO_agent
import numpy as np

device = PPO_agent.device
print("Using device: ",device)

env_name = './Reacher_Linux/Reacher.x86_64'
env = UnityEnvironment(file_name=env_name)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

policy = PPO_agent.Policy().to(device)

### Hyper-parameters
# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
# training loop max iterations
episode = 500
discount_rate = .99
epsilon = 0.1
beta = .01
tmax = 500
SGD_epoch = 4

# keep track of progress
mean_rewards = []

envs = env

for e in range(episode):

  # collect trajectories
  old_probs, states, actions, rewards = \
    PPO_agent.collect_trajectories(envs, policy, tmax=tmax)
  total_rewards = np.sum(rewards, axis=0)

  # gradient ascent step
  for _ in range(SGD_epoch):
    L = -PPO_agent.clipped_surrogate(policy, old_probs, states, actions, rewards,
                                      epsilon=epsilon, beta=beta)
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    del L

  # the clipping parameter reduces as time goes on
  epsilon *= .999

  # the regulation term also reduces
  # this reduces exploration in later runs
  beta *= .995

  # get the average reward of the parallel environments
  mean_rewards.append(np.mean(total_rewards))

  # display some progress every 20 iterations
  if (e + 1) % 20 == 0:
    print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
    print(total_rewards)

# save your policy!
torch.save(policy, './saved_models/PPO.policy')
