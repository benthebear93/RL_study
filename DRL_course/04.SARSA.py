import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output
%matplotlib inline
print ("plt version:[%s]"%(matplotlib.__version__))
print ("gym version:[%s]"%(gym.__version__))

'''
SARSA is a model-free algorithm which does not equire P where
Q value function should be estimated by samples.
The policy evaluation step for Q value function is

Q = sum[r + gamma*sum(Q*pi)]*P
where r + gamma*Q_old is TD target
(St, At, Rt+1, St+1, At+1) is needed

TD target is R + gamma Q
TD error is R + gamma Q - Q_old

for every time step
P.E
given SARSA
Q = Q + alpha*TD error

P.I
pi = eps/m + (1-eps)1(a = max Q(s, a'))
'''

class SARSAAgent():
    def __init__(self, n_state, n_action, epsilon=1.0, alpha=0.1, gamma=0.99):
        self.n_state = n_state
        self.n_action = n_action
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = np.zeros([n_state, n_action])
    
    def update_Q(self, state, action, reward, state_prime, action_prime, done):
        """
        Update Q value using TD learning
        """
        Q_old = self.Q[state][action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma*self.Q[state_prime][action_prime]
        td_error = td_target - Q_old
        # upate Q
        self.Q[state,action] = Q_old + self.alpha * td_error

    def update_epsilon(self, epsilon):
        self.epsilon = np.min([epsilon, 1.0]) #decay
    
    def get_action(self, state):
        if np.random.uniform() < self.epsilon: # random with epsilon probability 
            action = np.random.randint(0, high=self.n_action)
        else:
            action = np.argmax(self.Q[state])
        return action

# Run SARSA

env = gym.make('FrozenLake-v0')
n_state = env.observation_space.n
n_action = env.action_space.n
agent = SARSAAgent(n_state,n_action,epsilon=1.0,alpha=0.1,gamma=0.999)

# Loop
n_episode = 10000
for e_idx in range(n_episode):
    state = env.reset()
    action = agent.get_action(state)
    done = False
    while not done:
        state_prime,reward,done,info = env.step(action) # step 
        action_prime = agent.get_action(state_prime) # Get next action 
        agent.update_Q(state,action,reward,state_prime,action_prime,done) # online learning
        state = state_prime
        action = action_prime
    agent.update_epsilon(100/(e_idx+1))

E = np.zeros(shape=(4,4))
strs = ['S','F','F','F',
        'F','H','F','H',
        'F','F','F','H',
        'H','F','F','G',]
E[0,0] = 1 # Start
E[1,1]=E[1,3]=E[2,3]=E[3,0]=2 # Hole
E[3,3] = 3 # Goal
visualize_matrix(E,strs=strs,cmap='Pastel1',title='FrozenLake',fig_size=7)

# Plot Q
display_q_value(agent.Q,title="SARSA",
                fig_size=8,text_fs=8,title_fs=15)

gamma = 0.99
env = gym.make('FrozenLake-v0')
obs = env.reset() # reset
ret = 0
state = 0
for tick in range(1000):
    print("\n tick:[{}]".format(tick))
    env.render(mode='human')
    action = agent.get_action(state) # select action
    next_obs,reward,done,info = env.step(action)
    obs = next_obs
    ret = reward + gamma*ret 
    state = next_obs
    if done: break
env.render(mode='human')
env.close()
print ("Return is [{:.3f}]".format(ret))