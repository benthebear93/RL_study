import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import visualize
%matplotlib inline
import time

def value_iteration(env, gamma=0.99, eps=1e-6):
    # env information
    n_state = env.observation_space.n
    n_action = env.action_space.n

    # Transition probability
    # P[state][action] = [(prob, next state, reward, done), ...] 
    P = env.unwrapped.P 

    # Init value
    V = np.random.uniform(size=(n_state, 1)) # [n_state x1]

    # Loop
    tick, V_dists, V_list, Q_list = 0, [], [], []
    while True:
        tick = tick + 1
        Q = np.zeros(shape=(n_state, n_action)) # [n_state x n_action]
        for s in P.keys():  # for all states s
            for a in P[s].keys(): # for all actions a
                for prob, s_prime, reward, done in P[s][a]:
                    Q[s,a] += (reward + gamma*V[s_prime])*prob
        V_prime = np.max(Q, axis=1) # [n_state x 1]
        V_dist  = np.max(np.abs(V-V_prime))
        V_dists.append(V_dist)
        V_list.append(V)
        Q_list.append(Q)
        if V_dist < eps:
            break
    return Q,V,V_dists,V_list,Q_list

env = gym.make('FrozenLake-v0')
obs_space = env.observation_space
n_state = obs_space.n
action_space = env.action_space
n_action = action_space.n
print("Observation space:[{}]".format(n_state))
print("Action space:[{}]".format(n_action))
env.render()

M = np.zeros(shape=(4,4)
strs = ['S','F','F','F',
        'F','H','F','H',
        'F','F','F','H',
        'H','F','F','G',]
M[0,0] = 1 # Start
M[1,1]=M[1,3]=M[2,3]=M[3,0]=2 # Hole
M[3,3] = 3 # Goal
visualize_matrix(M,strs=strs,cmap='Pastel1',title='FrozenLake')

start = time.time()
Q,V,V_dists, V_list, Q_list = value_iteration(env, gamma=0.99, eps=1e-6)