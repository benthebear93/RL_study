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

# Run value iteration 
start = time.time()
Q,V,V_dists, V_list, Q_list = value_iteration(env, gamma=0.99, eps=1e-6)
print("It took [{:.2f}]s.".format(time.time()-start))
print("Value Iteration converged in [{}] loops.".format(len(V_dists)))

# Compute the optimal policy and plot
Pi = np.zeros((n_state, n_action))
Pi[np.arange(n_state), np.argmax(Q, axis=1)] = 1
plot_pi_v(Pi, np.reshape(V, (4,4)))

plt.plot(V_dists)
plt.title("COnvergence of Value Iteraction")
plt.show()

# Plot how the value function changes over iteration
n_plot = 5
for itr in np.round(np.linespace(0, len(Q_list)-1, n_plot)).astype(np.int32):
    V,Q = V_list[itr], Q_list[itr]
    Pi = np.zeros((n_state, n_action))
    Pi[np.`arange`(n_state), np.argmax(Q, axis=1)] = 1
    plot_pi_v(Pi, np.reshape(V, (4,4)), title="Value Function@iter={}".format(itr))

# Plot the value function of difference gammas
visualize_matrix(M, strs=strs, cmap='Pastel1', title='FrozenLake')
for gamma in [0.5, 0.9, 0.95, 0.99]:
    Q, V, V_dists, V_list, Q_list = value_iteration(env, gamma=gamma, eps=1e-6)
    Pi = np.zeros((n_state, n_action))
    Pi[np.arange(n_state), np.argmax(Q, axis=1)] = 1
    plot_pi_v(Pi, np.reshape(V, (4,4)), title='V (gamma:{:.2f})'.format(gamma))

# Run with the optimal policy
gamma = 0.99
Q, V, V_dists, V_list, Q_list = value_iteration(env, gamma=gamma, eps=1e-6)
Pi = np.zeros((n_state, n_action))
Pi[np.arange(n_state), np.argmax(Q, axis=1)] = 1
env = gym.make('FrozenLake-v0')
obs = env.reset()
ret = 0
for tick in range(1000):
    print("\n tick:[{}]".format(tick))
    env.render(mode='human')
    action = np.random.choice(n_action, 1, p=Pi[obs][:])[0] # optimal policy
    next_obs, reward, done, info = env.step(action)
    obs = next_obs
    ret = reward + gamma*ret
    if done : break
env.render(mode='human')
env.close()
print ("Return is [{:.3f}]".format(ret))

# Run with the random policy
env = gym.make('FrozenLake-v0')
obs = env.reset() # reset
ret = 0
for tick in range(1000):
    print("\n tick:[{}]".format(tick))
    env.render(mode='human')
    action = env.action_space.sample() # random action 
    next_obs,reward,done,info = env.step(action)
    obs = next_obs
    ret = reward + gamma*ret 
    if done: break
env.render(mode='human')
env.close()
print ("Return is [{:.3f}]".format(ret))