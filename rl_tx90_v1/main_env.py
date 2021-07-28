import gym
from gym.spaces import Discrete, Box, Tuple
import copy

import rospy
import roslaunch
import time


def distance(curr_pose, goal_pose):
	dist = np.sqrt(np.sum((curr_pose-goal_pose)**2, axis=0))
	return dist


class Tx90Env(gym.Env):
	def __init__(self):

		self.done = None
		self.reward = None
		self.max_steps = 1000 # limit the max episode step
		self.iterator = 0
		self.time_step = 0
		self.NrofJoints = 6

		self.x_min = -0.3
		self.x_max = 1.2
		self.y_min = -1.0
		self.y_max = 1.0
		self.z_min = 0.3
		self.z_max = 1.3
		self.distance_threshold = 0.01
		"""
		6 dof manipulator Actions:
			Type: Box(6)
			Num      Action         Min        Max 
			0        Joint1        -180        180
			1        Joint2        -130        147.5
			2        Joint3        -145        145
			3        Joint4        -270        270
			4        Joint5        -115        140
			5        Joint6        -270        270
		"""

		low  = np.deg2rad(np.array([-180,-130,-145,-270,-115,-270], dtype=np.float32))
		high = np.deg2rad(np.array([180, 147.5,145,270,140,270], dtype=np.float32))
		print("Action spaces: ",low,high)

		self.action_space = Box(low,high, dtype=np.float32)

				"""
		6 dof manipulator Observation:
			Type: Box(2)
			Num      Action         Min        Max 
			0        Joint1        -180        180
			1        Joint2        -130        147.5
			2        Joint3        -145        145
			3        Joint4        -270        270
			4        Joint5        -115        140
			5        Joint6        -270        270
		"""
		low = np.array([x_min, y_min, z_min], dtype=np.float32)
		high = np.array([x_max, y_max, z_max], dtype=np.float32)

		self.observation_space = Box(low,high, dtype=np.float32)

		self.tcp_pos = [0.0, 0.0, 0.0] # unit = meter
		self.tcp_ori = [0.0, 0.0, 0.0] # unit = meter

	def get_state(self, action):
		x,y,z,r,p,y = FK(action)

		self.state = ([x, y, z], [r, p, y])

		return self.state[0], self.state[1]

	def get_reward(self, goal):

		curr_pose = np.array(self.state[0])
		d = goal_distance(curr_pose, goal_pose)
		return d 

	def get_done(self):

		curr_pose = np.array(self.state[0])
		d = goal_distance(curr_pose, goal_pose)

		if self.time_step == self.max_steps:
			return True

		elif d < self.distance_threshold:
			return True

		else:
			return False

	def step(self, action, time_step):
		

	def reset(self):




