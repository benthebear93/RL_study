# 강화학습 ::  로봇 팔 openAI, Fetch env

```
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)
```
goal_a 와 goal_b 차이의 norm을 반환하는 함수이다. 
assert는 조건이 true가 아니면 AssertError를 발생시킨다. 
로봇팔의 경우 end_effector의 pos값과 target_pos의 차이에 대한 norm이 되겠다.
```python

class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)
```
먼저 FetchEnv라는 클래스에 robot_env.RobotEnv 를 상속시킨다. 
생성자의 경우  model_path, substeps, gripper_extra_height, block_gripper, has_object, target_in_the_air, target_offset, obj_range, target_range, distance_threshold, initial_qpos, reward_type을 받는다. 

몇가지는 당연한 내용이지만 헷갈리는 부분은 주석을 참고해보자.
block_gripper 는 gripper가 움직일 수 있는지 없는지에 대한 상태 표시 변수이다.
has_object는 환경에 object가 있는지 없는지에 대한 내용이다.
initial_qpos는 dictionary 형태로 선언된 조인트 이름들과 값들이다.
reward_type을 sparse로 할 지 dense로 할 지 고를 수 있게 해뒀다.

super로 생성자를 호출했다.  action이 4개이다.
```python
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
```
reward에 대한 계산이다. 
goal과 achieved goal의 거리를 계산하여 그 거리값을 reward로 주는데
sparse는 distance_threshold보다 클 경우에, dense일 경우는 그냥 distance값으로 준다. 
sparse와 dense reward에 대해서는 추후에 다뤄봐야겠다.
```python
def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()
```
정확히 뭘 위한 건지는 모르겠지만 gripper가 막혔을 경우 그리퍼의 joint를 초기화 한다.
```python
    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
```
set_action 함수는 4개로 받은 action을 설정해주는 함수이다.
action의 shape가 4인지 확인하고, 예외 상황을 대비하기 위해 action값을 복사해서 사용한다.
position control과 gripper control action을 만드는데 action[:3]과 action[3]이 있다. 선언 된 4개의 action 중에서 앞에 3개는 position control action, 1개는 gripper control이다.

position control에서 position 변화를 0.05로 제한했다.
rotation control은 fix된 상태이다. 툴(그리퍼)이 아래를 바라보는 형태로 고정
gripper control은 gripper_ctrl 2개를 array로 만들었다.

만약 그리퍼가 막혔으면 gripper control의 값을 0으로 만들어준다.
그리고 position control, rotation control, gripper control 배열을 concatenate로 합쳐준다.
```python
def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
```
observation을 가져오는 작업이다.
grip_pos와 velocity를 가져오고, dt의 경우 시뮬레이션의 timestep을 통해 가져온다.
그리고 robot_qpos와 qvel을 가져온다.

만약 object가 있다면, object의 위치와 rotation을 가져오고 velocity들도 가져온다.
grip 할때의 object의 상대적 위치, 속도도 가져온다. 
만약 object가 없다면 모두 0으로 설정해주면 된다.

추가로 만약 object가 없다면 achieved_goal은 gripper 위치가 되고 (gipper와 grip의 위치는 다르다)
있다면, object position을 받아오면 된다.

마지막으로 action처럼 observation도 concatenate를 이용해서 배열을 합쳐준다.
최종적으로 observation에는 그립할 때의 위치, 속도와 그리퍼의 상태, 속도를 포함하고 물체의 위치 상대위치 속도 상대 속도

rotation을 포함하고 있다.
```python

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return Tru
```
simulation 환경을 초기화 하는 것 같다.
initial_state를 확인해서 state들을 설정해준다.
object가 있다면 object의 시작 위치를 랜덤하게 설정해준다. 

```python
    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()
```
goal을 sample하는 함수 같다.
만약 object가 있다면 goal은 초기 그리퍼의 pos값들에서 uniform distribution을 가지는 값들을 추가한 값이 된다. 거기에 target_offset도 더해주고, z축의 경우 height offset도 더해준다. 

```python
    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
```
distance threshold에 들어올 경우 반환값으로 ...?
```python
def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
```
환경 셋업으로 initial_qpos값을 받아서 진행한다.
initial_qpos의 item을 참고해서 joint_qpos의 데이터들을 설정해준다.
end effector를 초기 위치로 옮긴다. 

끝!
