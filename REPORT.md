# Algorithm

We use a DDPG algorithm (Deep Deterministic Policy Gradient) ==>  hybrid method: Actor Critic. We use two neural networks:

a Critic that measures how good the action taken is (value-based). The value function maps each state action pair to a value which quantifies how is good to be / go to another state. The value function calculates what is the maximum expected future reward given a state and an action.

an Actor that controls how our agent behaves (policy-based). We directly optimize the policy without using a value function. This is useful when the action space is continuous or stochastic.

Instead of waiting until the end of the episode, we make an update at each step (TD Learning). The Critic observes our action and provides feedback in order to update our policy and be better at playing that game.

  
# Model architecture
The model is very simple and composed of 2 fully connected layer with leaky relu activation for each network.

* Actor model architecture
state => leaky_relu(FC1(state)) ==> tanh(leaky_relu(FC2(FC1))) ==> action<\br>
By applying a tanh function in output, we ensure that the action values are in the range [-1,1]

* Critc model architecture
state + action => leaky_relu(FC1(batchnorm(state)) ==> leaky_relu(FC2(FC1+action))) ==> value function <\br>


# Hyperparameters tuning
The most important parameter to tune is the learnin rate.
A basic grid search method can be applied to find the optimal value for the learning rate.

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


# Performance assessment

Solved Requirements: Considered solved when the average reward is greater than or equal to +30 over 100 consecutive trials.


  + Performance for DDPG Agent.
Environment is solved in 41 episodes. Average score: 30.10

![alt text](https://github.com/Adrelf/DRL_Continuous_Control/blob/master/images/Perfo_Reacher.png)


# Future improvements
  - Current algorithm: hyperparameters tuning with a grid search method
  - Policy gradient method:
    - PPO algorithm (WIP) ==> https://blog.openai.com/openai-baselines-ppo/
  - Actor critic method:
    - A2C or A3C ==> https://blog.openai.com/baselines-acktr-a2c/
