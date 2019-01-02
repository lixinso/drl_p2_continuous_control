
This project use DDPG (Deep Deterministic Policy Gradient) architecture.

## State and Action
The Agent is target to keep the position at the target location within as many time steps as possible. The reward is +0.1 for each step that the agent's hand is in the goal location.

The observation space consists of 33 variables corresponding to position, rotation, velocity and angular velocities of arm. The value must between -1 and 1.

## Algorithm

In the .ipython notebook, the ddpg function is used to train the agent. The training algorithm will stop if reached 3000 episodes or the average award is at least 30.0. Each episode continues until 100 timesteps is reached or done.

The reward for each step that the agent's hand is in the goal location is 0.1.

Neural Network is defined in ddpg_model.py. The Actor has 2 fully connected layers with 256 and 128 units, and RELU activation and tanh activation for action space. The Critic use 2 fully connected layers with 256 and 128 units, leaky_relu activation.

Continuous_Control.ipynb

ddpg_agent.py

- Agent: Interacts with and learns from the environment.
- class OUNoise: Ornstein-Uhlenbeck process.
- class ReplayBuffer: Fixed-size buffer to store experience tuples.

ddpg_model.py

- class Actor(nn.Module): Actor (Policy) Model. 
- class Critic(nn.Module): Critic (Value) Model.

## Hyper Paramters

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
N_LEARN_UPDATES = 10
N_TIME_STEPS = 20


## Result


## Future Improvement

- Try PPO (Proximal Policy OPtimization)
- Try D4PG (Distributed Distributional Deterministic Policy Gradients)
- Optimize computation speed with GPU 

## References
DDPG Bipedal
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal