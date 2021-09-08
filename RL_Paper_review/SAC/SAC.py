import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions.normal import Normal
"""
Study note that i will remove later
Basic understanding of Actor & Critic.

Old REINFORCE algorithm used Gt variable which can be used after episode is done.
So if we get a sample from Gt, we get high variance since it only update after end of the episode(trajectory)
To overcome this limitation, we replace the Gt wtih Q(s,a) that can be updated with various algorithms(TD, SARSA, etc).
With only st, at, we could extract sample before the episode ends. 
Summation in the equation still make variance exists but as we update Q value, we could minimize variance unlike REINFOCE.

Q will be parameterized as Qw(w as weight) which w are updated to minimize cost function.

With theta, updating P_theta(a_t|s_t) is ACTOR (Policy updating)

updating Qw(value function for action) is Critic 

what are the main key about SAC then?

SAC uses a entropy concepts that makes learning stable.
expectation of entropy of policy about state marginal is added to objective to consider maximum entropy obejctive.
Policy will try to maximize entropy and expected return.
H as entropy measure, alpha as a temperature parameter. 
maximizing entropy will make policy "explore more" and take a look at near-optimal strategy. 
focus on Policy theta, Soft Qw, state-value V

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done): #state, #new state_ 
        index  = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch] 

        return states, actions, rewards, states_, dones

class Actor(nn.Module):
    def __init__(self, learning_rate):
        super(Actor, self).__init__()
        self.fc1

class SAC:
    def __init__(self, state_dim, action_dim, action_bounds, offset, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256, hidden_size=256, update_interval=1, buffer_size=int(1e6), target_entropy=None):
        super().__init__()

        #actor
        self.policy_net = Actor(state_dim, action_dim, action_bounds, offset).to(device) 
        #critic
        self.value_net = Critic(state_dim, action_dim).to(device)
        self.Q_net = Q(state_dim, action_dim).to(device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
    
    def select_action(self, state, staet)