import gym
import numpy as np
from gym import spaces

class Tx90Env(gym.Env):
	def __init__(self):
		self.tcp_pos = [0.0, 0.0, 0.0] # unit = meter
		self.x_range = [-0.3, 1.2] # range of x
		self.y_range = [-1.0, 1.0] # range of y
		self.z_range = [ 0.3, 1.3] # range of z
		self.tcp_ori = []

