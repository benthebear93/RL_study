import gym
import numpy as np

from gym import spaces

print("import complete")

class GridEnv(gym.Env):
	"""
	Description :
	16x16 grid with agent. Star is where agent should go, 
	block mark is forbided to go. 


	Source : UNIST-LIM-LAB/DRL-course

	Observation: 
		Type: Box(12)
		Num      Observation                Min          Max
		0        distance to wall(up)        0            16(15)
		1        distance to wall(down)      0            16(15)
		2        distance to wall(right)     0            16(15)
		3        distance to wall(left)      0            16(15)
		4        distance to wall(up)        0            16(15)
		5        distance to wall(down)      0            16(15)
		6        distance to wall(right)     0            16(15)
		7        distance to wall(left)      0            16(15)
		8        distance to wall(up)        0            16(15)
		9        distance to wall(down)      0            16(15)
	   10        distance to wall(right)     0            16(15)
	   11        distance to wall(left)      0            16(15)
	   12        distance to wall(up)        0            16(15)
	   13        distance to wall(down)      0            16(15)
	   14        distance to wall(right)     0            16(15)
	   15        distance to wall(left)      0            16(15)
	Action:
		Type: Discrete(4)
		NUm   Action
		0     move up
		1     move down
		2     move left
		3     move right
	Reward:
		Finding star reward 1, restrict position reward -1

	Episode Termination:
		moving out the map, stepping on the restrict position or star end the game (done)
	"""

    def __init__(self):
        self.world_shape = [16, 16] # 16x16 gridworld
        self.agent_pos = [1, 0] # init pos
        self.restrict_pos = [0, 1] # resticted pos
        self.star_pos = [0, 3] # goal 
        self.action_space = spaces.Discrete(4) # (위, 아래, 왼쪽, 오른쪽) 4가지 action
        #self.obs_space = spaces.Discrete(3) # (빈 곳, 금지표시, 별) 3가지 observation
        self.obs_space = spaces.Tuple((spaces.Discrete(12),spaces.Box(low=0, high=16, shape=1))
        #tuple Discrete for 12
        # 1,2,3,4 for distance to wall
        # 5,6,7,8    for distance to stop
        # 9,10,11,12 for distance to star

    def step(self, action):
        if action == 0:
            self.agent_pos[0] -= 1 #up 
        elif action == 1: 
            self.agent_pos[0] += 1 #down 
        elif action == 2: 
            self.agent_pos[1] -= 1 #left
        elif action == 3: 
            self.agent_pos[1] += 1 #right
        else:
            raise Exception("the action is not defined")
    
        info = {}

        return self._get_obs(), self._get_reward(), self._is_done(), info

    def reset(self): #reset the world , GUI need to added here
        self.world = np.zeros(self.world_shape)
        self.world[0, 1] = 1 # 금지표시
        self.world[0, 3] = 2 # 별
        self.agent_pos = [1, 0]

        return self._get_obs()

    def render(self): # render GUI
        self.world[self.agent_pos[0], self.agent_pos[1]] = -1
        print(self.world)
        self.world[self.agent_pos[0], self.agent_pos[1]] = 0

    def close(self):
        pass
 
    def _get_obs(self): # get observation 
        if self.agent_pos == [1, 1]:
            return 1 # 금지표시 아래
        elif self.agent_pos == [1, 3]:
            return 2 # 별 아래
        else:
            return 0 # 빈 곳
    
    def _get_reward(self): # get reward
        # 금지표시
        if self.agent_pos == self.restrict_pos:
            return -1
        # 별 cases
        elif self.agent_pos == self.star_pos:
            return 1
        # 빈곳
        else:
            return 0
    
    def _is_done(self):
        # 맵 밖으로 나갔을 때
        if self.agent_pos[0] < 0 or self.agent_pos[0] > 1:
            return True
        if self.agent_pos[1] < 0 or self.agent_pos[1] > 3:
            return True
        # 금지표시
        if self.agent_pos == self.restrict_pos:
            return True
        # 별
        if self.agent_pos == self.star_pos:
            return True
        # 나머지
        else:
            return False

env = GridEnv()

for i_episode in range(20):
    obs = env.reset()
    for t in range(100):
        env.render()
        print(obs)
        # action = env.action_space.sample() # random action
        action = 3 # Move right
        obs, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
