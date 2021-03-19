
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    """with out this, agent might goes with only one direction 
    due to the numpy only returning first value"""
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(
    id='FrozenLake-v4',
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name':'4x4','is_slippery':False})

env = gym.make("FrozenLake-v4")
env.render()

Q = np.zeros([env.observation_space.n, env.action_space.n])
# space 16 , action 4 , initalize Q with 0 
num_episodes = 2000
# 2000 times running

rList =[]
for i in range(num_episodes):
    state = env.reset() # reset the env, and bring the first state
    rAll = 0
    done = False
    while not done: # check if game is not done or not
        action = rargmax(Q[state, :]) #select action, if Q is same, use random argmax
        new_state, reward, done, _ = env.step(action) #new state and reward by action

        Q[state, action] = reward + np.max(Q[new_state, :]) #update Q-table
        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate : " + str(sum(rList)/num_episodes))
print("Final Q table values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()