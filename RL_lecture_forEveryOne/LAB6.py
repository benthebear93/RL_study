import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import tensorflow as tf

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': True}
)

env = gym.make('FrozenLake-v3')

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

X = tf.Variable(tf.ones(shape=[1,input_size], dtype = tf.float32))
W = tf.Variable(tf.random_uniform([input_size, output_size],0,0.01))

Qpred = tf.matmul(X,W)
Y = tf.Variable(tf.ones(shape=[1,output_size],dtype =tf.float32))

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

dis =.99
num_episodes = 2000

rList = []

def one_hot(x):
    return np.identity(16)[x:x+1]

for i in range(num_episodes):
    s = env.reset()
    e = 1./((i/50)+10)
    rAll = 0
    done = False
    local_loss = []

    while not done:
        Qs = sess.run(Qpred, feed_dict={X:one_hot(s)})
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = reward
        else:
            Qs1 = sess.run(Qpred, feed_dict={X:one_hot(s1)})
            Qs[0, a] = reward + dis * np.max(Qs1)

        sess.run(train, feed_dict={X:one_hot(s), Y:Qs})
        rAll += reward
        s = s1
    List.append(rAll)

