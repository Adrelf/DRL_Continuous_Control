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

agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

def ddpg(n_episodes=2000, max_t=1000):
  scores_mean = deque(maxlen=100)  # last 100 scores
  scores = []
  best_score = 0
  best_average_score = 0
  for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations  # get the initial state
    agent.reset()
    scores_agents = np.zeros(num_agents)
    for t in range(max_t):
      action = agent.act(state)
      env_info = env.step(action)[brain_name]
      next_state = env_info.vector_observations  # get the next state
      reward = env_info.rewards  # get the reward
      done = env_info.local_done  # see if episode has finished
      agent.step(state, action, reward, next_state, done, t)
      state = next_state
      scores_agents += reward
      if np.any(done):
        break
    score = np.mean(scores_agents)
    scores_mean.append(score)
    average_score = np.mean(scores_mean)
    scores.append(score)
    if score > best_score:
      best_score = score
    if average_score > best_average_score:
      best_average_score = average_score
    print(
      "Episode:{}, Low Score:{:.2f}, High Score:{:.2f}, Average Score:{:.2f}, Best Avg Score:{:.2f}".format(
        i_episode, scores_agents.min(), scores_agents.max(), average_score, best_average_score))
    if average_score >= 30.0:
      print(
        '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, average_score))
      torch.save(agent.actor_local.state_dict(), './saved_models/checkpoint_actor.pth')
      torch.save(agent.critic_local.state_dict(), './saved_models/checkpoint_critic.pth')
      break
    if i_episode % 100 == 0:
      torch.save(agent.actor_local.state_dict(), './saved_models/checkpoint_actor.pth')
      torch.save(agent.critic_local.state_dict(), './saved_models/checkpoint_critic.pth')
      print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))
  return scores


scores = ddpg(n_episodes=2000, max_t=1000)
env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

plt.savefig('./figures/training_performance.png')



