import gym
from gym.envs.registration import register
import readchar


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}


register(
    id='FrozenLake-v4',
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={'map_name':'4x4','is_slippery':False})


'''여기서부터 gym 코드의 시작이다. env 는 agent 가 활동할 수 있는 environment 이다.'''

env = gym.make("FrozenLake-v4")
env.render() #환경을 화면으로 출력

while True:
    key = readchar.readkey()  #키보드 입력을 받는다

    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key] #에이젼트의 움직임
    state, reward, done, info = env.step(action) #움직임에 따른 결과값들
    env.render() #화면을 다시 출력
    print("State:", state, "Action", action, "Reward:", reward, "Info:", info)

    if done: #도착하면 게임을 끝낸다.
        print("Finished with reward", reward)
        break