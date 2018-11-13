from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from brain.DDPG_agent import Agent

env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')

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

env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)

agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)

def ddpg(n_episodes=2000, max_t=700):
  scores_window = deque(maxlen=100)  # last 100 scores
  scores = []
  max_score = -np.Inf
  for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]  # get the initial state
    agent.reset()
    score = 0
    for t in range(max_t):
      action = agent.act(state)
      env_info = env.step(action)[brain_name]
      next_state = env_info.vector_observations[0]  # get the next state
      reward = env_info.rewards[0]  # get the reward
      done = env_info.local_done[0]  # see if episode has finished
      agent.step(state, action, reward, next_state, done)
      state = next_state
      score += reward
      if done:
        break
    scores_window.append(score)
    scores.append(score)
    print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_window), score), end="")
    if i_episode % 100 == 0:
      torch.save(agent.actor_local.state_dict(), './saved_models/checkpoint_actor.pth')
      torch.save(agent.critic_local.state_dict(), './saved_models/checkpoint_critic.pth')
      print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    if np.mean(scores_window) >= 30.0:
      print(
        '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
      torch.save(agent.actor_local.state_dict(), './saved_models/checkpoint_actor.pth')
      torch.save(agent.critic_local.state_dict(), './saved_models/checkpoint_critic.pth')
      break
  return scores


scores = ddpg(n_episodes=2000, max_t=500)
env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()



