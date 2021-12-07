import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym.envs.toy_text.cliffwalking import CliffWalkingEnv
'''
Have one Q function Q
Givne (S,A,R,S)
Find the best action S_t+1
a* = argmax Qold(S_t+1, a)

Update Q function with (S,A,R,S,a*) with 1 step TD error
Qnew = Qold + alpha(R+gamma*Q(S,a*)-Qold(S,A))

Policy improve
Behavior policy pi(a|s)=epsilon/m + (1-epsilon)1 (a=max Q(s,a'))
Target policy pi(a|s) =1 (a=max Q(s,a'))
'''
class Q_learning():
    def __init__(self,n_state,n_action,alpha=0.5,epsilon=1.0,gamma=0.999):
        self.n_state = n_state
        self.n_action = n_action
        self.alpha_init = alpha
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Initial Q value
        self.Q = np.zeros([n_state,n_action])
        
    def update_Q(self,state,action,reward,state_prime,done):
        """
        Update value
        """
        Q_old = self.Q[state][action]
        # TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma*np.max(self.Q[state_prime])
        td_error = td_target - Q_old # TD error
        self.Q[state,action] = Q_old + self.alpha*td_error # update Q
        
    def update_epsilon(self,epsilon):
        self.epsilon = np.min([epsilon,1.0]) 
        
    def update_alpha(self,alpha):
        self.alpha = np.min([alpha,self.alpha_init]) 
        
    def get_action(self,state):
        """
        Get action
        """
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0,high=self.n_action)
        else:
            action = np.argmax(self.Q[state])
        return action 

'''
Double Q learning 
Have two Q function Qa, Qb
Given (S,A,R,S)
Qa = Qa + alpha(R+gamma*Qb(S_t+1, a*)-Qa(S, A))

Policy improve
pi = epsilon/m +(1-epsilon)1 (a=max Qa(s,a'))
Target policy = pi = 1(a=max Qa(s,a'))
'''

class Double_Q_learning():
    def __init__(self,n_state,n_action,alpha=0.5,epsilon=1.0,gamma=0.999):
        self.n_state = n_state
        self.n_action = n_action
        self.alpha_init = alpha
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Initialize Two Q tables
        self.Q_A = np.zeros([n_state,n_action])
        self.Q_B = np.zeros([n_state,n_action])

    def update_Q(self,state,action,reward,state_prime,done):
        """
        Update value
        """
        if np.random.uniform() < 0.5:
            Q_update = self.Q_A
            Q_target = self.Q_B
        else:
            Q_update = self.Q_B
            Q_target = self.Q_A

        action_prime = np.argmax(Q_update[state_prime])

        # TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * Q_target[state_prime][action_prime]

        Q_old = Q_update[state][action]
        td_error = td_target - Q_old # TD error
        Q_update[state,action] = Q_old + self.alpha * td_error # update Q

    def update_epsilon(self,epsilon):
        self.epsilon = np.min([epsilon,1.0]) 
        
    def update_alpha(self,alpha):
        self.alpha = np.min([alpha,self.alpha_init]) 
        
    def get_action(self,state):
        """
        Get action
        """
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0,high=self.n_action)
        else:
            action = np.argmax(self.Q_A[state])
        return action 

env = CliffWalkingEnv()
env.render()
print ("Number of states:", env.nS)
print ("Number of actions that an agent can take:", env.nA)
action_space = ["up", "right", "down", "left"]
print ("Current state", env.s)
print ("Transitions from current state:", env.P[env.s])
for s in range(48):
    for a in range(4):
        if env.P[s][a][0][2] == -100:
            env.P[s][a][0] = (env.P[s][a][0][0], env.P[s][a][0][1], env.P[s][a][0][2], True)

import random

def train(env, agent, n_episode):
    env.reset()
    is_terminal = False
    # Loop
    for e_idx in range(n_episode):
        state = env.reset() # reset environment, select initial 
        action = agent.get_action(state)
        done = False
        while not done:
            state_prime, reward, done, info = env.step(action) # step 
            action_prime = agent.get_action(state_prime) # Get next action
            agent.update_Q(state, action, reward, state_prime, done) # learns Q
            state = state_prime
            action = action_prime
        agent.update_epsilon(100/(e_idx+1)) # Decaying epsilon
    print ("Training Done.")

q_learning = Q_learning(n_state=48,n_action=4,epsilon=1.0,alpha=0.1,gamma=0.999)
double_q_learning = Double_Q_learning(n_state=48,n_action=4,epsilon=1.0,alpha=0.1,gamma=0.999)

train(CliffWalkingEnv(), q_learning, 300)
train(CliffWalkingEnv(), double_q_learning, 300)

from time import sleep

def run(env, agent):
    is_terminal = False
    state = env.reset()
    env.render()
    while not is_terminal:
        action = agent.get_action(state)
        print("Action taken:", action_space[action])
        next_state, reward, is_terminal, t_prob = env.step(action)
        print("Transition probability:", t_prob)
        print("Next state:", next_state)
        print("Reward recieved:", reward)
        print("Terminal state:", is_terminal)
        state = next_state
        env.render()
        sleep(1)
