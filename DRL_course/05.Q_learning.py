import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from visualize import *
'''
to implement model-free policy iteration, we need to estimate the following using samples:
Qk+1 = sum[ r(s,a,s')+gamma*max Q(s',a')]P(s'|s,a)
update Q value using TD target and error
Qnew(S, A) <- Qold(S,A) + alpha(R+gamma*max Q(S, a')-Qold(S, A))
Update estimator online, (S_t,A_t, R_t+1,S_t+1) is needed , not A_t+1
TD target is R_t+1 + gamma*maxQ_old(St+1, a')
TD error is R_t+1 + gamma*maxQ_old(St+1, a')-Q(S_t, A_t)

Algorithm 
Policy eval
Collect (S,A,R,S)
TD target = R_t+1 + gamma*maxQ_old(St+1, a')
TD error = gamma*maxQ_old(St+1, a')-Q(S_t, A_t)
Q[S,A] = Q[S,A]+alpha*TD target

policy improvement
pi = epsilon/m + (1-epsilon)1 (alpha=maxQ(s,a'))
'''

class Q_learningAgent():
    def __init__(self, n_state, n_action, alpha=0.5, epsilon=1.0, gamma=0.999):
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
    

env = gym.make('FrozenLake-v0')
n_state = env.observation_space.n
n_action = env.action_space.n
# epsilon=1.0 -> random policy
agent = Q_learningAgent(n_state,n_action,epsilon=1.0,alpha=0.1,gamma=0.999)
# Loop 
n_episode = 3000
for e_idx in range(n_episode):
    state = env.reset()
    action = agent.get_action(state)
    done = False
    while not done:
        state_prime, reward, done, info = env.step(action)
        action_prime = agent.get_action(state_prime)
        agent.update_Q(state, action, reward, state_prime, done)
        state = state_prime
        action = action_prime
        
    # agent.update_epsilon(1000/(e_idx+1)) # reduce randomness
    agent.update_alpha(1000/(e_idx+1)) # reduce update rate
print ("Q-learning done.")

# Plot env
E = np.zeros(shape=(4,4))
strs = ['S','F','F','F',
        'F','H','F','H',
        'F','F','F','H',
        'H','F','F','G',]
E[0,0] = 1 # Start
E[1,1]=E[1,3]=E[2,3]=E[3,0]=2 # Hole
E[3,3] = 3 # Goal

visualize_matrix_Q(E,strs=strs,cmap='Pastel1',title='FrozenLake',fig_size=7)
display_q_value(agent.Q,title="Q Learning",fig_size=8,text_fs=9,title_fs=15)    

