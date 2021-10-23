import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time
from visualize import *

class MCAgent():
    def __init__(self,n_state,n_action,epsilon=1.0,alpha=0.1,gamma=0.995):
        """
        Initialize Monte Carlo Learning Agent 
        """
        self.n_state = n_state
        self.n_action = n_action
        self.epsilon = epsilon # epsilon greediness
        self.alpha = alpha # Q mixing rate
        self.gamma = gamma # discount factor
        
        # Initialize Q value
        self.Q = np.zeros([n_state,n_action])
        # Memory 
        self.samples = []

    def svae_sample(self, state, action, reward, done):
        """
        Save experience (s,a, r, done)
        """
        self.samples.append([state, action, reward, done]) 

    def update_Q(self):
        """
        Update Q
        """
        Q_old = self.Q # [S x A]
        g = 0
        G = Q_old
        for t in reversed(range(len(self.smaples))):
            state,action,reward,_ = self.sample[t]
            g = reward + self.gamma*g # g = r + gamma * g
            G[state][action] = g
        # update Q
        self.Q = Q_old + self.alpha*(G -Q_old)
        self.samples = []

    def update_epsilon(self,epsilon):
        self.epsilon = np.min([epsilon,1.0]) # decay

    def get_action(self,state):
        """
        Get action
        """
        if np.random.uniform() < self.epsilon: # random with epsilon probability 
            action = np.random.randint(0, high=self.n_action)
        else: # greedy action
            action = np.argmax(self.Q[state])
        return action

class ImportanceSampling(MCAgent):
    def __init__(self, *args, **kwargs):
        super(ImportanceSampling, self).__init__(*args, **kwargs)
        self.C = np.zeros([self.n_state,self.n_action])
        self.b = self.set_behavior_policy(p_type='random')

    def set_behavior_policy(self, p_type):
        """
        Set behavior policy
        """
        if p_type == "random":
            def behavior_policy(state):
                prob = np.ones(self.n_action) / self.n_action
                return prob
        elif p_type == "greedy":
            def behavior_policy(state):
                prob = np.zeros(self.n_action)
                prob[np.argmax(state)] =1
                return prob
        return behavior_policy

    def update_Q(self):
        """
        Update Q
        """
        g = 0.
        for t in reversed(range(len(self.samples))):
            state,action,reward,_ = self.samples[t]
            g = (reward+self.gamma*g) # g = (r + gamma * g) * 1/b
            self.C[state][action] += 1
            self.Q[state][action] += (1 / self.C[state][action]) * (g - self.Q[state][action])
            if action !=  np.argmax(self.Q[state]):
                break
                # Empty memory
        self.samples = []

    def get_action(self, state):
        """
        Get action
        """
        probs = self.b(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        return action

class WeightedImportanceSampling(ImportanceSampling):
    def __init__(self, *args, **kwargs):
        super(WeightedImportanceSampling, self).__init__(*args, **kwargs)
          
    def update_Q(self):
        """
        Update Q
        """
        g = 0.
        W = 1.
        for t in reversed(range(len(self.samples))): # for all samples in a reversed way
            state,action,reward,_ = self.samples[t]
            g = reward + self.gamma*g # g = r + gamma * g
            self.C[state][action] += W            
            self.Q[state][action] += (W / self.C[state][action]) * (g - self.Q[state][action])
            if action !=  np.argmax(self.Q[state]):
                break
            W = W * 1./self.b(state)[action]
            
        # Empty memory
        self.samples = []
                   
def q_policy_evaluartion(env, P, r, Pi, gamma=0.99, epsilon=1e-6):
    """
    Policy evaluation 
     env   : environment
     P     : state transition probability [S x A x S]
     r     : reward [S]
     Pi    : policy [S x A]
    returns
     Q     : Q-value [S x A]
    """
    # env info
    n_state = env.observation_space.n
    n_action = env.action_space.n
    # Randomly init Q
    Q = np.random.uniform(size=(n_state,n_action))
    while True:
        V = np.sum(Pi*Q, axis=1) # [S]
        V_tile = np.tile(V[np.newaxis, np.newaxis,:], reps=(n_state,n_action, 1)) # [S x A x S]
        Q_prime = np.sum((r+gamma*V_tile)*P, axis=2) # [S x A]
        Q_dist = np.max(np.max(np.abs(Q-Q_prime)))
        Q = Q_prime
        if Q_dist < epsilon:
            break
    return Q

def q_policy_improvement(env, Q):
    """
    Policy improvement
     env   : environment
     Q     : Q-value [S x A]
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    Pi = np.zeros((n_state,n_action))
    Pi[np.arange(n_state),np.argmax(Q,axis=1)] = 1 # Greedy policy update
    return Pi

def q_policy_iteration(env, gamma=0.99, epsilon=1e-6):
    """
    Policy iteration
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    # Random init policy
    Pi = np.random.uniform(size=(n_state,n_action))
    Pi = Pi/np.sum(Pi,axis=1,keepdims=True)
    # Parse P and r from env
    P = np.zeros((n_state, n_action, n_state))
    r = np.zeros((n_state, n_action, n_state))
    for s in env.unwrapped.P.keys(): # for all states s
        for a in env.unwrapped.P[s].keys(): # for all actions a
            for prob,s_prime,reward,done in env.unwrapped.P[s][a]:
                P[s][a][s_prime] = prob # model ??
                r[s][a][s_prime] = reward
    while True:
        Q = q_policy_evaluartion(env, P, r, Pi, gamm=gamma, epsilon=epsilon)
        Pi_prime = q_policy_improvement(env, Q)
        if (Pi==Pi_prime).all():
            break
        Pi = Pi_prime
    return Pi, Q

env = gym.make('FrozenLake-v0')
obs = env.reset()
Pi,Q = q_policy_iteration(env)
display_q_value(Q, title="Q Function",fig_size=8,text_fs=9,title_fs=15)

# run MC learning
n_state = env.observation_space.n
n_action = env.action_space.n
M = MCAgent(n_state,n_action,epsilon=1.0,alpha=0.1,gamma=0.999)
# Loop
n_episode = 10000
for e_idx in range(n_episode):
    state = env.reset()
    action = M.get_action(state)
    done = False
    while not done:
        next_state, reward, done, info = env.state(action)
        if done:
            if reward==0: reward =-5
            else: reward =+5
        else:
            reward = -0.0
        next_action = M.get_action(next_state)
        M.svae_sample(state,action, reward, done)
        state = next_state
        action = next_action
    M.update_Q()
    M.update_epsilon(100/(e_idx+1))
print("MC Learning done")

E = np.zeros(shape=(4,4))
strs = ['S','F','F','F',
        'F','H','F','H',
        'F','F','F','H',
        'H','F','F','G',]
E[0,0] = 1 # Start
E[1,1]=E[1,3]=E[2,3]=E[3,0]=2 # Hole
E[3,3] = 3 # Goal

# Plot env
visualize_matrix(E,strs=strs,cmap='Pastel1',title='FrozenLake',fig_size=7)
# Plot Q
display_q_value(M.Q,title="Monte Carlo Policy Iteration", 
                fig_size=8,text_fs=8,title_fs=15)

M = ImportanceSampling(n_state,n_action,epsilon=1.0,alpha=0.1,gamma=0.999)

# Loop
n_episode = 10000
for e_idx in range(n_episode):
    state = env.reset() # reset environment, select initial 
    action = M.get_action(state)
    done = False
    while not done:
        next_state, reward, done, info = env.step(action) # step 
        if done:
            # Reward modification to handle sparse reward
            if reward == 0:  reward = -5 # let hole to have -1 reward
            else: reward = +5 # let goal to have +10 reward
        else:
            reward = 0.0 # constant negative reward
        next_action = M.get_action(next_state) # Get next action
        M.save_sample(state,action,reward,done) # Store samples
        state = next_state
        action = next_action
    # End of the episode
    M.update_Q() # Update Q value using sampled epsiode
print ("MC learning done.")
# Plot env
visualize_matrix(E,strs=strs,cmap='Pastel1',title='FrozenLake',fig_size=7)
# Plot Q
display_q_value(M.Q,title="Monte Carlo Policy Iteration", 
                fig_size=8,text_fs=8,title_fs=15)