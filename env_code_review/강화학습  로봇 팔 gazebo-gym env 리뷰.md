# 강화학습 :: 로봇 팔 gazebo-gym env 리뷰

link : https://github.com/erlerobot/gym-gazebo

gym은 간단하게 말해서 강화학습 알고리즘을 돌릴 수 있는 Environment , 환경 구축을 도와주는 프레임워크이다.agent야 내가 구현한다지만 일반적인 환경을 구성해야 state, reward 등을 설정할 수 있기 때문에 gym 환경 구축 꼭 필요한 부분이다. 

```python

import gym
class SimpleGymEnv(gym.Env):
    def __init__(self):
        # Some initialization for Env
        pass

    def step(self, action):
        obs, reward, done, info = 0, 0, True, {}
        return obs, reward, done, info

    def reset(self):
        obs = 0
        return obs

    def render(self):
        # Code for visualization
        pass

    def close(self):
        # Clear env
        pass

```

**gym class**

 **__init__**에서는 환경을 구성하는 변수들을 선언해준다.

observation_msg, scale, x_index, obs, reward, done, reward_dist, reward_ctrl, max_episode_steps 등 그 외에 환경 하이퍼 파라미터라고 하여 선언 된 것 들이 있다. 이유는 모르겠지만,

target 위치를 pos와 ori로 선언했다. 끝단의 velocity도 선언되어 있다. 로봇의 joint position도 선언한다. publisher와 subscriber를 위한 토픽도 선언한다. 각 조인트의 이름들을 선언하였고, 각 링크의 이름들도 하이퍼 파라미터로 선언하였다 

```python
        EE_POS_TGT = np.asmatrix([0.3305805, -0.1326121, 0.4868]) # center of the H in the SCARA robot
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES = np.asmatrix([[0, 0, 0]])
        # Initial joint position
        INITIAL_JOINTS = np.array([2.0, -2.0, -2.0, 0.])
        # Used to initialize the robot, #TODO, clarify this more
        # STEP_COUNT = 2  # Typically 100.
        slowness = 10000000 # 1 is real life simulation
        # slowness = 1 # use >10 for running trained network in the simulation
        # slowness = 10 # use >10 for running trained network in the simulation

        # Topics for the robot publisher and subscriber.
        JOINT_PUBLISHER = '/articulated_arm_controller/command'
        JOINT_SUBSCRIBER = '/articulated_arm_controller/state'
        # joint names:
        # joint names:
        MOTOR1_JOINT = 'motor1'
        MOTOR2_JOINT = 'motor2'
        MOTOR3_JOINT = 'motor3'
        MOTOR4_JOINT = 'motor4'

        # Set constants for links
        BASE = 'base_link'

        HANS_MOTOR = 'articulated_arm_hans'
        HANS_SUPPORT = 'arm_link'
        HANS_LINK = 'arm_link_arm'
        HANS_ADAPTER = 'arm_link_paladapter'

        PAL_MOTOR = 'pal_motor'
        PAL_MOTOR2 = 'pal_motor2'
        PAL_MOTOR_SUPPORT= 'pal_motor_adapter'
        PAL_LINK = 'pal_link_adapter'
        PAL_ADAPTER = 'adapter_hebi'

        HEBI_MOTOR = 'hebi_black'
        HEBI_MOTOR2 = 'hebi_red'
        HEBI_SUPPORT = 'hebi_adapter'
        HEBI_ADAPTER = 'end_effector_adapter'

        EE_LINK = 'ee_link'

        JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT, MOTOR4_JOINT]
        LINK_NAMES = [BASE,
                        HANS_MOTOR,     HANS_SUPPORT,       HANS_LINK, HANS_ADAPTER,
                        PAL_MOTOR,        PAL_MOTOR2,     PAL_MOTOR_SUPPORT,  PAL_LINK,  PAL_ADAPTER,
                        HEBI_MOTOR,       HEBI_MOTOR2,    HEBI_SUPPORT,                  HEBI_ADAPTER,
                        EE_LINK]

```

reset_condition이라는 것도 딕셔너리 형태로 선언되어 있는데 어디에 사용되는지는 추후에 알아보자.

```python
        reset_condition = {
            'initial_positions': INITIAL_JOINTS,
             'initial_velocities': []
        }

```
위에 선언 된 것들을 사용해서 self.environment를 선언한다
```python
        self.environment = {
            # rk changed this to for the mlsh
            # 'ee_points_tgt': ee_tgt,
            'ee_points_tgt': self.realgoal,
            'joint_order': m_joint_order,
            'link_names': m_link_names,
            'slowness': slowness,
            'reset_conditions': reset_condition,
            'tree_path': URDF_PATH,
            'joint_publisher': m_joint_publishers,
            'joint_subscriber': m_joint_subscribers,
            'end_effector_points': EE_POINTS,
            'end_effector_velocities': EE_VELOCITIES,
        }

```
신기한점은 tree_path라고 해서 URDF_PATH를 가져온다는 거다. 어디에 쓰이는지는 추후에..
```
        #   note that the xacro of the urdf is updated by hand.
        # The urdf must be compiled.
        _, self.ur_tree = treeFromFile(self.environment['tree_path'])
        # Retrieve a chain structure between the base and the start of the end effector.
        self.scara_chain = self.ur_tree.getChain(self.environment['link_names'][0], self.environment['link_names'][-1])
        # print("nr of jnts: ", self.scara_chain.getNrOfJoints())
        # Initialize a KDL Jacobian solver from the chain.
        self.jac_solver = ChainJntToJacSolver(self.scara_chain)
        #print(self.jac_solver)
        self._observations_stale = [False for _ in range(1)]
        #print("after observations stale")
        self._currently_resetting = [False for _ in range(1)]
        self.reset_joint_angles = [None for _ in range(1)]

```
가져온 urdf를 tree형태로 가져오고 이를 다시 chain 형태로 가져온다. 여기서는 기본 kinematic solver인 KDL을 사용하는 걸로 하는데 KDL에는 getChain 함수가 있다. IKFast에도 GetChain이 있긴하다. 이 부분은 추후에 고민해야할 듯. KDL에서 자코비안 solver를 가져온다.
```python
self.obs_dim = self.scara_chain.getNrOfJoints() + 6
```
observation 차원을 설정하는 과정에서 로봇팔은
자유도 + 끝단의 위치 + 끝단의 속도를 가져와서 총 12차원이 된다.
여기서 위치가 단순히 position인 것 같은데 나는 orientation도 고려해줘야하니깐 차원을 늘리던지 아니면 속도를 없애던지 해야겠다.

```python
        low = -np.pi/2.0 * np.ones(self.scara_chain.getNrOfJoints())
        high = np.pi/2.0 * np.ones(self.scara_chain.getNrOfJoints())
        # print("Action spaces: ", low, high)
        self.action_space = spaces.Box(low, high)
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

```
action space에 대한 선언을 위해 low, high를 설정해준다. 이때 로봇 자체의 control range를 가져올 수 있어야하는데 그렇지 못한다고 한다. 이 부분이 joint limit을 의미하는 건지, 아니면 다른걸 의미하는지 확인해봐야겠다. observation도 선언되었다. 

```python
    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg =  message

```
JointTrajectoryControllerState의 subscriber의 callback 인 것 같다.
```python
 def get_trajectory_message(self, action, robot_id=0):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """
        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = self.environment['joint_order']
        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        action_float = [float(i) for i in action]
        target.positions = action_float
        # These times determine the speed at which the robot moves:
        # it tries to reach the specified target position in 'slowness' time.
        target.time_from_start.secs = self.environment['slowness']
        # target.time_from_start.nsecs = self.environment['slowness']
        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]
        return action_msg

```
Joint anlge을 JointTrajectory로 선언하여 publish하는데 사용하는 것 같다. 그리고 각 point들을 만들어서 로봇한테 움직이도록 보내준다. 잘 이해가 가지는 않는다 추후에 돌아오는 걸로
```python
    def process_observations(self, message, agent, robot_id=0):
        """
        Helper fuinction to convert a ROS message to joint angles and velocities.
        Check for and handle the case where a message is either malformed
        or contains joint values in an order different from that expected observation_callback
        in hyperparams['joint_order']
        """
        if not message:
            print("Message is empty");
            # return None
        else:
            # # Check if joint values are in the expected order and size.
            if message.joint_names != agent['joint_order']:
                # Check that the message is of same size as the expected message.
                if len(message.joint_names) != len(agent['joint_order']):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER

                # Check that all the expected joint values are present in a message.
                if not all(map(lambda x,y: x in y, message.joint_names,
                    [self._valid_joint_set[robot_id] for _ in range(len(message.joint_names))])):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER
                    print("Joints differ")
            return np.array(message.actual.positions) # + message.actual.velocities

```
로스에서 혹은 가제보를 통한 로스에서 들어오는 observation 값들을 위에 선언한 하이퍼파라미터 형태로바꿔주려는 작업이다. 
```python
    def get_jacobians(self, state, robot_id=0):
        """
        Produce a Jacobian from the urdf that maps from joint angles to x, y, z.
        This makes a 6x6 matrix from 6 joint angles to x, y, z and 3 angles.
        The angles are roll, pitch, and yaw (not Euler angles) and are not needed.
        Returns a repackaged Jacobian that is 3x6.
        """
        # Initialize a Jacobian for 6 joint angles by 3 cartesian coords and 3 orientation angles
        jacobian = Jacobian(self.scara_chain.getNrOfJoints())
        # Initialize a joint array for the present 6 joint angles.
        angles = JntArray(self.scara_chain.getNrOfJoints())
        # Construct the joint array from the most recent joint angles.
        for i in range(self.scara_chain.getNrOfJoints()):
            angles[i] = state[i]
        # Update the jacobian by solving for the given angles.observation_callback
        self.jac_solver.JntToJac(angles, jacobian)
        # Initialize a numpy array to store the Jacobian.
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(jacobian.rows())])
        # Only want the cartesian position, not Roll, Pitch, Yaw (RPY) Angles
        ee_jacobians = J
        return ee_jacobians

```
Joint angle에서 x,y,z를 매핑해주는 자코비안을 구한다. 자코비안을 정확히 왜 구하는지는 잘 모르겠다. 
뭔가 플래닝에서인가 로봇의 자세? 싱귤러리티?에 대한 걸 확인할 때 사용했던 것 같은데 정확하지 않다.
```python
    def get_ee_points_jacobians(self, ref_jacobian, ee_points, ref_rot):
        """
        Get the jacobians of the points on a link given the jacobian for that link's origin
        :param ref_jacobian: 6 x 6 numpy array, jacobian for the link's origin
        :param ee_points: N x 3 numpy array, points' coordinates on the link's coordinate system
        :param ref_rot: 3 x 3 numpy array, rotational matrix for the link's coordinate system
        :return: 3N x 6 Jac_trans, each 3 x 6 numpy array is the Jacobian[:3, :] for that point
                 3N x 6 Jac_rot, each 3 x 6 numpy array is the Jacobian[3:, :] for that point
        """
        ee_points = np.asarray(ee_points)
        ref_jacobians_trans = ref_jacobian[:3, :]
        ref_jacobians_rot = ref_jacobian[3:, :]
        end_effector_points_rot = np.expand_dims(ref_rot.dot(ee_points.T).T, axis=1)
        ee_points_jac_trans = np.tile(ref_jacobians_trans, (ee_points.shape[0], 1)) + \
                                        np.cross(ref_jacobians_rot.T, end_effector_points_rot).transpose(
                                            (0, 2, 1)).reshape(-1, self.scara_chain.getNrOfJoints())
        ee_points_jac_rot = np.tile(ref_jacobians_rot, (ee_points.shape[0], 1))
        return ee_points_jac_trans, ee_points_jac_rot

    def get_ee_points_velocities(self, ref_jacobian, ee_points, ref_rot, joint_velocities):
        """
        Get the velocities of the points on a link
        :param ref_jacobian: 6 x 6 numpy array, jacobian for the link's origin
        :param ee_points: N x 3 numpy array, points' coordinates on the link's coordinate system
        :param ref_rot: 3 x 3 numpy array, rotational matrix for the link's coordinate system
        :param joint_velocities: 1 x 6 numpy array, joint velocities
        :return: 3N numpy array, velocities of each point
        """
        ref_jacobians_trans = ref_jacobian[:3, :]
        ref_jacobians_rot = ref_jacobian[3:, :]
        ee_velocities_trans = np.dot(ref_jacobians_trans, joint_velocities)
        ee_velocities_rot = np.dot(ref_jacobians_rot, joint_velocities)
        ee_velocities = ee_velocities_trans + np.cross(ee_velocities_rot.reshape(1, 3),
                                                       ref_rot.dot(ee_points.T).T)
        return ee_velocities.reshape(-1)

```
```python
    def get_ee_points_velocities(self, ref_jacobian, ee_points, ref_rot, joint_velocities):
        """
        Get the velocities of the points on a link
        :param ref_jacobian: 6 x 6 numpy array, jacobian for the link's origin
        :param ee_points: N x 3 numpy array, points' coordinates on the link's coordinate system
        :param ref_rot: 3 x 3 numpy array, rotational matrix for the link's coordinate system
        :param joint_velocities: 1 x 6 numpy array, joint velocities
        :return: 3N numpy array, velocities of each point
        """
        ref_jacobians_trans = ref_jacobian[:3, :]
        ref_jacobians_rot = ref_jacobian[3:, :]
        ee_velocities_trans = np.dot(ref_jacobians_trans, joint_velocities)
        ee_velocities_rot = np.dot(ref_jacobians_rot, joint_velocities)
        ee_velocities = ee_velocities_trans + np.cross(ee_velocities_rot.reshape(1, 3),
                                                       ref_rot.dot(ee_points.T).T)
        return ee_velocities.reshape(-1)

```
잘 모르겠다. 한가지 알겠는 건 결국 ee의 속도나 jacobian을 구하는 부분은 내가 짜야된다는 거다(?)
take_observation이라고 해서, 현재 환경의 observation값을 return 해주는 함수다. 앞서 선언한 process_observation에 observation message와 환경을 넣어줘서 last_observation을 받는다.
그리고 끝단부의 jacobian을 구한다. 구한 걸 Homogeneous matrix로 만들어서 (여기서는 왜 rotation matrix라고 선언했지?) quaternion을 구한다. (근데 안쓰는 변수다)
끝단의 현재 위치, 속도, 마지막 observation을 구해서 return 값으로 보낸다.
```python
  def take_observation(self):
        """
        Take observation from the environment and return it.
        TODO: define return type
        """
        # Take an observation
        # done = False

        obs_message = self._observation_msg
        if obs_message is None:
            # print("last_observations is empty")
            return None

        # Collect the end effector points and velocities in
        # cartesian coordinates for the process_observationsstate.
        # Collect the present joint angles and velocities from ROS for the state.
        last_observations = self.process_observations(obs_message, self.environment)
        # # # Get Jacobians from present joint angles and KDL trees
        # # # The Jacobians consist of a 6x6 matrix getting its from from
        # # # (# joint angles) x (len[x, y, z] + len[roll, pitch, yaw])
        ee_link_jacobians = self.get_jacobians(last_observations)
        if self.environment['link_names'][-1] is None:
            print("End link is empty!!")
            return None
        else:
            # print(self.environment['link_names'][-1])
            trans, rot = forward_kinematics(self.scara_chain,
                                        self.environment['link_names'],
                                        last_observations[:self.scara_chain.getNrOfJoints()],
                                        base_link=self.environment['link_names'][0],
                                        end_link=self.environment['link_names'][-1])
            # #
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = rot
            rotation_matrix[:3, 3] = trans
            # angle, dir, _ = rotation_from_matrix(rotation_matrix)
            # #
            # current_quaternion = np.array([angle]+dir.tolist())#

            # I need this calculations for the new reward function, need to send them back to the run scara or calculate them here
            current_quaternion = quaternion_from_matrix(rotation_matrix)
            current_ee_tgt = np.ndarray.flatten(get_ee_points(self.environment['end_effector_points'],
                                                              trans,
                                                              rot).T)
            ee_points = current_ee_tgt - self.realgoal#self.environment['ee_points_tgt']
            ee_points_jac_trans, _ = self.get_ee_points_jacobians(ee_link_jacobians,
                                                                   self.environment['end_effector_points'],
                                                                   rot)
            ee_velocities = self.get_ee_points_velocities(ee_link_jacobians,
                                                           self.environment['end_effector_points'],
                                                           rot,
                                                           last_observations)

            # Concatenate the information that defines the robot state
            # vector, typically denoted asrobot_id 'x'.
            state = np.r_[np.reshape(last_observations, -1),
                          np.reshape(ee_points, -1),
                          np.reshape(ee_velocities, -1),]

            return np.r_[np.reshape(last_observations, -1),
                          np.reshape(ee_points, -1),
                          np.reshape(ee_velocities, -1),]

```
step 함수는 action에 대한 env의 상태 혹은 결과를 알려준다. 있어야할 것들은 reward, done, action, observation등 단순히 reward 를 구현해서 done을 확인하고 trajectory(action)를 publish 하면된다
```python
 def step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - reward
            - done (status)
            - action
            - observation
            - dictionary (#TODO clarify)
        """
        self.iterator+=1
        self.reward_dist = -self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])

        # here we want to fetch the positions of the end-effector which are nr_dof:nr_dof+3
        if(self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])<0.005):
            self.reward = 1 - self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)]) # Make the reward increase as the distance decreases
            print("Reward is: ", self.reward)
        else:
            self.reward = self.reward_dist
        done = (bool(abs(self.reward_dist) < 0.005)) or (self.iterator > self.max_episode_steps)
        self._pub.publish(self.get_trajectory_message(action[:self.scara_chain.getNrOfJoints()]))
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()

        # Return the corresponding observations, rewards, etc.
        # TODO, understand better what's the last object to return
        return self.ob, self.reward, done, {}

```
이외에도 함수들이 몇개 더 있는데 이정도가 메인인 것 같다.
생각보다 엄청 어렵지는 않을 것 같으면서 막상 내가하려고 하면 어려울 것 같다. 약간 수치해석기반 Inverse Kinematic한 느낌이다.

