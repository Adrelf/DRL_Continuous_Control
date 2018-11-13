# Algorithm
DDPG (actor critic method).

  
# Model architecture
The model is very simple and composed of 2 fully connected layer with relu activation for each actor.


# Hyperparameters tuning
The most important parameter to tune is the learnin rate.
A basic grid search method can be applied to find the optimal value for the learning rate.

Hyperparamters
  - learning rate of the actor: 1e-3
  - learning rate of the critic: 3e-4
  - optimizer: Adam
  - L2 weight decay: 0.0001
  - batch size: 128
  - discount factor: 0.99
  - soft update of target parameters: 1e-3
  - replay buffer siz: 1e6
  - model architecture: FC [256,128] for the actor, FC [256,128] for the critic


# Performance assessment

Solved Requirements: Considered solved when the average reward is greater than or equal to +30 over 100 consecutive trials.


  + Performance for DDPG Agent.
Environment not solved. Average score after 2000 episodes: 10.62

![alt text](https://github.com/Adrelf/DRL_Continuous_Control/blob/master/images/Perfo_Reacher.png)


# Future improvements
  - Current algorithm: hyperparameters tuning with a grid search method
  - Policy gradient method:
    - PPO algorithm (WIP) ==> https://blog.openai.com/openai-baselines-ppo/
  - Actor critic method:
    - A2C or A3C ==> https://blog.openai.com/baselines-acktr-a2c/
