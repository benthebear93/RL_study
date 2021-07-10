import gym
import numpy as np
from gym import spaces
from tkinter import *

class GridEnv(gym.Env):
	def __init__(self):
		self.world_shape = [16, 16]
		self.agent_pos = [1, 0]
		self.restrict_pos = [0, 1]
		self.star_pos = [0, 3]
		self.action_space = spaces.Discrete(4) # up down right left 
		self.obs_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4), spaces.Discrete(4))) #distance from wall, forbidden, star as Tuple

	def step(self, action):
		if action == 0:
			self.agent_pos[0] -= 1
		elif action == 1:
			self.agent_pos[0] += 1
		elif action == 2:
			self.agent_pos[1] -= 1
		elif action == 3:
			self.agent_pos[1] += 1
		else:
			raise Exception("the action is not defined")

		info = {}

		return self._get_obs(), self._get_reward(), self._is_done(), info

	def reset(self):
		self.world = np.zeros(self.world_shape)
		restrict_x = np.random.randint(0,15)
		restrict_y = np.random.randint(0,15)
		self.world[restrict_y, restrict_y] = 1
		star_y = np.random.randint(0,15)
		star_y = np.random.randint(0,15)
		self.world[star_y, star_y] = 2
		agent_x = np.random.randint(0,15)
		agent_y = np.random.randint(0,15)
		self.agent_pos=[agent_y, agent_x]

	def render(self):
		self.world[self.agent_pos[0], self.agent_pos[1]] = -1
		print(self.world)
		self.world[self.agent_pos[0], self.agent_pos[1]] = 0

	def close(self):
		pass

	def _get_obs(self):
		if self.agent_pos == [1, 1]:
			return 1 #at forbidden
		elif self.agent_pos == [1, 3]:
			return 2 #at star
		else:
			return 0 #at empty

	def _get_reward(self):
		if self.agent_pos == self.restrict_pos: #forbidden
			return -1
		elif self.agent_pos == self.star_pos:
			return 1
		else:
			return 0

	def _is_done(self):
		if self.agent_pos[0] < 0 or self.agent_pos[0] > 1: #out of the map
			return True
		if self.agent_pos[1] < 0 or self.agent_pos[1] > 3:
			return True
		if self.agent_pos == self.restrict_pos:
			return True
		if self.agent_pos == self.star_pos:
			return True
		else:
			return False

env = GridEnv()
# root =Tk()
# root.mainloop()
for i_episode in range(20):
	obs = env.reset()
	print("reset")
	for t in range(100):
			env.render()
			#print(obs)
			# action = env.action_space.sample() # random action
			action = 3 # Move right
			obs, reward, done, info = env.step(action)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
			break

env.close()