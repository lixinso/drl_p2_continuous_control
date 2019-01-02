
[image1]: ./result.png "ResultPlot"

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

Agent Parameters

- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 128        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR_ACTOR = 1e-4         # learning rate of the actor 
- LR_CRITIC = 1e-4        # learning rate of the critic
- WEIGHT_DECAY = 0        # L2 weight decay
- N_LEARN_UPDATES = 10
- N_TIME_STEPS = 20


## Result

<p>
Episode 1	 Score: 0.00	Average Score: 0.000.00
Episode 2	 Score: 0.00	Average Score: 0.000.00
Episode 3	 Score: 0.03	Average Score: 0.010.03
Episode 4	 Score: 0.00	Average Score: 0.010.00
Episode 5	 Score: 0.14	Average Score: 0.030.14
...
Episode 997	 Score: 27.94	Average Score: 20.93.94
Episode 1200	 Score: 31.56	Average Score: 25.5556
Episode 1200	 Average Score: 25.55
Episode 1201	 Score: 27.89	Average Score: 25.6089
...
Episode 1275	 Score: 33.28	Average Score: 29.8528
Episode 1276	 Score: 35.48	Average Score: 29.8948
Episode 1277	 Score: 34.48	Average Score: 29.9848
Episode 1278	 Score: 30.41	Average Score: 30.0541
Environment solved in 1178 episodes. 	 Average Score: 30.05
</p>

![ResultPlot][image1]

## Future Improvement

- Try PPO (Proximal Policy OPtimization)
- Try D4PG (Distributed Distributional Deterministic Policy Gradients)
- Optimize computation speed with GPU 

## References
DDPG Bipedal
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal