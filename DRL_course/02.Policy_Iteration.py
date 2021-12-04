import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time
from visualize import *

def policy_evaluation(env, Pi, gamma=0.99, epsilon=1e-6):
    # Policy evaluation
    '''
    Compute the value function for a fixed policy pi
    Following bellam expectation eq for Vpi is used
    '''
    # Extract environment information
    obs_space = env.observation_space
    n_state = obs_space.n
    P = env.unwrapped.P
    # Random initial value function 
    V = np.random.uniform(size=(n_state,1))
    # Loop
    tick,V_dists,V_list = 0,[],[]
    while True:
        tick = tick + 1
        V_prime = np.zeros((n_state,))
        for s in P.keys(): # for all state
            for a in P[s].keys(): # for all actions
                for prob,s_prime,reward,done in P[s][a]: 
                    V_prime[s] += (reward+gamma*V[s_prime])*prob*Pi[s][a]
        V_dist = np.max(np.abs(V-V_prime))
        V_dists.append(V_dist)
        V = V_prime
        V_list.append(V)
        if V_dist < epsilon:
            break
    return V,V_dists,V_list

def policy_improvement(env, V, gamma=0.99):
    """
    Policy Improvement
    In this step, update the policy distribution using the value function
    computed by policy evaluation. A new policy distribution is updated
    greedly. 
    """
    obs_space = env.observation_space
    n_state = obs_space.n
    act_space = env.action_space
    n_action = act_space.n
    P = env.unwrapped.P
    Q = np.zeros((n_state,n_action))
    # Loop
    for s in P.keys(): # for all states
        for a in P[s].keys(): # for all actions
            for prob, s_prime, reward, done in P[s][a]:
                Q[s,a] += (reward + gamma*V[s_prime])*prob  
    Pi = np.zeros((n_state, n_action))
    Pi[np.arange(n_state), np.argmax(Q, axis=1)] = 1
    return Pi

def policy_iteration(env, gamma=0.99, epsilon=1e-6):
    """
    Policy Iteration
    """
    obs_space = env.observation_space
    n_state = obs_space.n
    act_space = env.action_space
    n_action = act_space.n
    Pi = np.random.uniform(size=(n_state, n_action))
    Pi = Pi/np.sum(Pi, axis=1, keepdims=True)
    while True:
        V,V_dist,V_list = policy_evaluation(env, Pi, gamma=gamma, epsilon=epsilon)
        Pi_prime = policy_improvement(env, V, gamma=gamma)
        if (Pi == Pi_prime).all():
            break
        Pi = Pi_prime
    return Pi, V

# ENV INIT
env = gym.make('FrozenLake-v0')
obs_space = env.observation_space
n_state = obs_space.n
action_space = env.action_space
n_action = action_space.n

print("Observation space:[{}]".format(n_state))
print("Action space:[{}]".format(n_action))

# initial policy
Pi = np.random.uniform(size=(n_state,n_action))
Pi = Pi/np.sum(Pi,axis=1,keepdims=True)
np.set_printoptions(precision=3,suppress=True)
plot_policy(Pi,title='Initial Policy')

# RUn policy evaluation
start = time.time()
V, V_dists, V_list = policy_evaluation(env, Pi, gamma=0.99, epsilon=1e-6)
print("It took [{:.2f}]s.".format(time.time()-start))
print ("Policy evaluation converged in [{}] loops.".format(len(V_dists)))

plt.plot(V_dists)
plt.title("Convergence of Policy Evaluation")
plt.show()

n_plot = 5
for itr in np.round(np.linspace(0,len(V_list)-1,n_plot)).astype(np.int32):
    V = V_list[itr]
    plot_pi_v(Pi,np.reshape(V,(4,4)),cmap='binary',title="Value Function@iter={}".format(itr))

# Env
visualize_matrix(M, strs=strs, cmap='Pastel1', title='FrozenLake')

# Initial policy
Pi = np.random.uniform(size=(n_state,n_action))
Pi = Pi/np.sum(Pi,axis=1,keepdims=True)
np.set_printoptions(precision=3,suppress=True)
plot_policy(Pi,title='Initial Policy')

# Run value iteration
V,V_dists,V_list = policy_evaluation(env,Pi,gamma=0.99,epsilon=1e-6)
plot_pi_v(Pi,np.reshape(V,(4,4)),
          title='Value Function',title_fs=15,REMOVE_TICK_LABELS=True)

# Run policy improvement
Pi = policy_improvement(env,V)
plot_pi_v(Pi,np.reshape(V,(4,4)),title="Policy Improvement")

# Run the optimal Policy , Initialize environment
obs = env.reset() # reset
ret,gamma = 0,0.99
for tick in range(1000):
    print("\n tick:[{}]".format(tick))
    env.render(mode='human')
    action = np.random.choice(n_action,1,p=Pi[obs][:])[0] # select action
    next_obs,reward,done,info = env.step(action)
    obs = next_obs
    ret = reward + gamma*ret 
    if done: break
env.render(mode='human')
env.close()
print ("Return is [{:.3f}]".format(ret))




