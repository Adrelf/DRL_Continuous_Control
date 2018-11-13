from parallelEnv import parallelEnv
import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# this is useful for batch processing especially on the GPU
def preprocess_batch(images):
  batch_input = torch.from_numpy(images).float().unsqueeze(0).to(device)
  return batch_input


# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(envs, policy, tmax=200, nrand=5, action_size=4):
  brain_name = envs.brain_names[0]
  brain = envs.brains[brain_name]
  # reset the environment
  env_info = envs.reset(train_mode=True)[brain_name]
  # number of parallel instances
  n = len(env_info.agents)

  # initialize returning lists and start the game!
  state_list = []
  reward_list = []
  prob_list = []
  action_list = []


  # start all parallel agents
  envs.step([1] * action_size * n)

  # perform nrand random steps
  for _ in range(nrand):
    action = np.random.randn(action_size, n)
    action = np.clip(action,-1,1)
    env_info = envs.step(action)[brain_name]
    fr1 = env_info.vector_observations[0]  # get the next state
    re1 = env_info.rewards[0]  # get the reward

  for t in range(tmax):

    # prepare the input
    batch_input = preprocess_batch(fr1)

    # probs will only be used as the pi_old
    # no gradient propagation is needed
    # so we move it to the cpu
    action, probs, entropy = policy(batch_input)
    action = np.array(action)
    # we take one action and skip game forward
    env_info = envs.step(action)[brain_name]
    fr1 = env_info.vector_observations[0]  # get the next state
    re1 = env_info.rewards[0]  # get the reward
    is_done = env_info.local_done[0]  # see if episode has finished
    reward = re1

    # store the result
    state_list.append(batch_input)
    reward_list.append(reward)
    prob_list.append(probs)
    action_list.append(action)

    # stop if any of the trajectories is done
    # we want all the lists to be retangular
    if n> 1:
      if is_done.any():
        break
    else:
      if is_done:
        break


  # return pi_theta, states, actions, rewards, probability
  return prob_list, state_list, \
         action_list, reward_list


# convert states to probability, passing through the policy
def states_to_prob(policy, states):
  states = torch.stack(states) # Dim - [tmax,n_env,state_size]
  policy_input = torch.reshape(states, (states.shape[0]*states.shape[1], states.shape[2]))
  return policy(policy_input)


# clipped surrogate function
# similar as -policy_loss for REINFORCE, but for PPO
def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):
  discount = discount ** np.arange(len(rewards))
  rewards = np.asarray(rewards) * discount #[:, np.newaxis]

  # convert rewards to future rewards
  rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

  mean = np.mean(rewards_future, axis=0)
  std = np.std(rewards_future, axis=0) + 1.0e-10

  rewards_normalized = (rewards_future - mean) / std


  # convert everything into pytorch tensors and move to gpu if available
  actions = torch.tensor(actions, dtype=torch.float, device=device)
  rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

  # convert states to policy (or probability)
  action, log_prob, entropy = states_to_prob(policy, states)

  # ratio for clipping
  ratio = (log_prob[:,0] - tensor(old_probs)).exp()
  # clipped function
  clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
  clipped_surrogate_value = torch.min(ratio*rewards, clip*rewards)
  # Compute entropy ==> - sum(p(x).log(p(x)))
  policy_loss = torch.min(clip, clipped_surrogate_value).mean(0) + beta * entropy.mean()
  return policy_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
  fan_in = layer.weight.data.size()[0]
  lim = 1. / np.sqrt(fan_in)
  return (-lim, lim)

def tensor(x):
  if isinstance(x, torch.Tensor):
    return x
  x = torch.tensor(x, dtype=torch.float32)
  return x

class Policy(nn.Module):
  def __init__(self, state_size=33, action_size=4, seed=0, fc1_units = 256, fc2_units=128):
    super(Policy, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, fc1_units)
    self.fc2 = nn.Linear(fc1_units, fc2_units)
    self.fc3 = nn.Linear(fc2_units, action_size)
    self.reset_parameters()
    self.std = nn.Parameter(torch.zeros(1, action_size))

  def reset_parameters(self):
    self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    self.fc3.weight.data.uniform_(-3e-3, 3e-3)

  def forward(self, state, action=None):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x_out = F.tanh(F.relu(self.fc3(x)))
    dist = torch.distributions.Normal(x_out, F.softplus(self.std))
    if action is None:
      action = dist.sample()
    else:
      action = tensor(action)
    log_prob = dist.log_prob(action)
    log_prob = torch.sum(log_prob, dim=1, keepdim=True)
    entropy = torch.sum(dist.entropy(), dim=1, keepdim=True)
    return action, log_prob, entropy
