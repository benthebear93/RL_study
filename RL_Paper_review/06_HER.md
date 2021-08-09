**[Abstract]**

sparse reward(희소 리워드)은 RL에서 가장 어려운 문제중에 하나이다. 논문에서는 리워드 엔지니어링을 복잡하게 하지 않아도 희소하거나 이진화된 리워드에서 sample-efficient learning을 하는 방법인 HER에 대해서 이야기한다. 



논문에서는 주로 로봇 팔로 물체를 움직이는 작업에 대해서 다룬다. 이때 보상은 이진형태로 목적을 달성 했는지 안했는지 만 고려한다.

https://www.youtube.com/watch?v=Dz_HuzgMxzo

<iframe src="https://www.youtube.com/embed/Dz_HuzgMxzo?feature=oembed" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" width="200" height="113" frameborder="0"></iframe>

**[Introduction]**

RL이 NN과 합쳐지면서 순차적으로 선택을 해야되는 문제에 대한 성공이 가능하게 됐다. 하지만 리워드 함수에 대한 엔지니어링은, 특히 로봇 관련해서, 꼭 필요했다. 특히 벽돌을 쌓아올리는 작업에 대한 policy를 학습시키기 위해 5개의 복잡한 항들을 조합해서 cost function을 만들어 가중치를 조절해야만 가능했다. 이런 보상 엔지니어링은 RL에 대한 지식과 실제 시스템(로봇)에 대한 도메인 지식 둘 다 필요하기 때문에 RL을 실제 환경에 적용하는데 큰 방해가 된다. 따라서 리워드에 대한 엔지니어링이 없어도 잘 학습할 수 있는 알고리즘을 개발하는게 필요하다.



사람은 원하지 않는 행동을 통해서도 학습할 수 있다. 하지만 현재 RL은 그렇지 않다. 예를 들어 사람은 하키 퍽으로 점수를 내는데 "오른쪽으로 살짝 치우쳐서 골을 넣지 못했다"라면 여기서 학습하여 다음번에는 살짝 왼쪽으로 치려고 한다는 것이다. 

논문에서는 알고리즘이 위와 같은 이성적인 행동을 하도록 HER라는 기술을 소개하며 이는 어떤 off-policy도 적용할 수 있다.



HER은 달성할 수 있는 "다수의" goal이 있다면 적용될 수 있다. 각 시스템의 state를 도달하는 것은 각각 다른 goal로 볼 수 있다. 

HER은 sample 효율을 증가시키는 것 뿐아니라 리워드 신호가 희소하거나 이진적이라도 학습을 가능하게 한다. 논문에서 제안하는 방법은 universal 한 policy를 학습하는 데 근본을 두고 있고, 이는 현재 state만 input으로 받는 것이 아니라 goal state도 받는다. 메인 아이디어는 다른 agent가 도달하려는 goal말고 다른 goal을 도달하는 에피소드를 replay 하는 것 같다. 



**[Hindsight Experience Replay]**

**3.1 A motivating example**

state space와 action space가 아래와 같은 bit flipping env(reward가 spare인 환경) 를 고려해보자. 

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA4MDlfMjEz/MDAxNjI4NDcxMTY3MTQw.aQCe5ydKSoz78RGA2OH-yiL9arkQEUf5xxRompcSC_8g.pqdH4FwRzXqo_LCrvtT_19nXVw0-piOOnpg_fA9GEiwg.PNG.nswve/Screenshot_from_2021-08-09_10-05-56.png?type=w966)                                                                    

모든 episode에서 시작 state와 도달 state를 uniform하게 sample한다. policy는 target state에 도달하지 못하면 reward -1을 받는다.

일반적인 RL 알고리즘은 위와 같은 환경에 n이 40 이상일 경우 학습에 실패하기 마련이다. 이는 reward를 -1 말고는 받지 못하기 때문이다.

이때 improving exploration, count-based exploration 같은 테크닉을 사용해도 이런 환경에서는 도움이 안되는데, 문제가 state를 얼마나 다양하게 방문하냐가 아니기 때문이다. 문제는 단순히 너무 큰 state space를 explore하는 것은 비효율적이기 때문이다. 



일반적인 해결방법으로는 agent를 도달할 수 있게 reward를 바꿔주는 방법이다. 다만 이는 복잡한 문제에서는 도입하기 어렵다. 



reward shaping말고 도메인 지식이 필요없는 방법을 논문에서는 제시한다.

근간이 되는 아이디어는 기존에 만들어진 trajectory를 다른 goal에 대해 **재평가** 하는 것이다. 

기존 trajectory가 **s1,,,,,sT**를 돌았고 sT가 g(goal)은 아니었어도, 이 trajectory는 적어도 sT에 도달하는 방법에 대해서는 알려줄 수 있다.

이런 정보는 off-policy RL을 통해서 얻어질 수 있고 replay buffer에 있는 g를 sT로 바꿔 replay를 진행할 수 있다. 



추가로 기존 g도 replay buffer에 남겨서 실행 할 수 있다.

이런 식의 변화로 replay된 trajectory가 적어도 -1은 아닌 reward를 갖게 되어 학습이 조금 더 간단해진다.



**3.2 Multi-goal RL**

논문에서는 agent가 여러 다른 goal을 학습하길 원했다. Universal Function Approximator 논문에서 착안하여 policy와 value function을 state와 goal을 input으로 주어 학습시켰다. 심지어 한가지 goal만을 가지고 학습시키는 것보다 다수의 goal을 가지고 학습시키는게 더 쉽다는 걸 보인다. 



논문에서는 몇가지 가정을 한다.

가정 1. 모든 goal은 몇 predicate fg S:->{0,1}과 연관이 있고 agent의 목적은 fg(S)=1이 되게 하는 state s를 찾는것을 목적으로 한다.

가정  2. 주어진 state s에서 도달할 수 있는 goal g 를 쉽게 찾을 수 있다.



Universal policy는 arbitrary RL 알고리즘을 통해 학습될 수 있다. 이는 특정 분포에서 초기 state와 goal을 sampling하는 것으로 진행 되는데, goal에 도착하지 않았을 때는 음수 보상을 주는 방법으로 timestep마다 agent를 돌린다. reward function이 희소하기 때문에 이 방법은 잘 성공하지 못한다. 이를 해결하기 위해서 논문에서는 HER를 소개한다. 



**3.3 Algorithm**

HER의 기본적인 아이디어는 매우 간단하다. 몇번의 에피소드(s0->sT)를 경험한다음 매 transition(sT->sT+1) 마다 replay buffer에 넣는다. 이때 단순히 에피소드에서 사용 된 Original goal만이 아니라 다른 goal들의 집합도 같이 넣어준다. 이때 goal 자체는 agent의 action의 영향에 의해 추구되기 때문에 환경의 다이나믹스랑은 연관이 없다. 따라서 각 trajectory를 임의의 goal에 대해서 replay할 수 있다. (DQN과 같은 off-policy 알고리즘을 사용한다는 가정하에) 



HER을 사용하기 위해서는 replay를 위한 추가적인 goal들을 설정해줘야한다. 가장 간단한 알고리즘으로 보자면,에피소드의 final state에서 얻어진 goal과 함께 trajectory를 replay 했다. 논문에서는 replay를 위한 추가적인 goal을 다른 타입과 질(quantities)을 비교하여 실험하였다. 또한 모든 경우에서 각 에피소드에서 원하는 original goal도 같이해서 trajectory를 replay 했다.



자세한 이해를 위해 알고리즘 수도 코드를 확인해보자.

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA4MDlfMTU4/MDAxNjI4NDcxMjE2MDYw.lSO5PW4DsGQNp70TgSQ1zaurAhIE8EAMBeiFHQtUSQkg.ZofRm32mzgm0-BDVMAYBh9_bDR1dkoefDVVckcEwy8cg.PNG.nswve/Screenshot_from_2021-08-09_10-06-42.png?type=w966)                                                                    

에피소드마다 policy를 따라 움직이는데, state와 goal을 같이 넣어주고 중간에 추가적인 goal을 뽑아서 다시 한번 돌려준다.

이때 추가적인 goal에 대한 reward와 transition을 replay buffer에 저장한다. 논문에서 이야기하는 실패한 경우에도 reward를 받아서 sparse reward를 해결하려는 방법으로 추가적인 goal을 만들어 reward를 저장하는 모습을 볼 수 있다. 



HER 알고리즘은 어쩌면 한가지 goal에서 자연스럽게 replay에서 사용되는 goal로 넘어가는 암시적 curriculum의 형태라고 볼 수 있다. 이때 goal은 랜덤한 agent한테도 도달하기 쉬운 곳부터 더 어려운 곳으로 바뀐다고 볼 수 있다. 하지만 명시적 curriculum과 반대로 HER은 초기 환경의 분포를 컨트롤할 필요가 전혀 없다. HER가 매우 극단적인 희소 보상으로 부터 학습하지만, 실험적으로 봤을 때 보상이 잘 만들어진 상태에서보다 더 좋은 성능을 보였다. 이런 결과는 보상을 좋게 짜는 것에 대한 실질적인 문제를 시사하며, 잘 짜여진 reward는 종종 우리가 진짜 신경쓰는 metric에 타협으로 이루어지곤 한다(?)



**[Experiments]**

**4.1 Environments**

7자유도 로봇팔 , Fetch 사, Mujoco 사용

Policy들은 Multi-Layer Perceptrons과 Rectified Linear Unit 활성화 함수로 표현됐다. 

DDPG+Adam으로 학습이 진행됐으며, 8 worker를 통해 매 update마다 파라미터를 평균내는 형태로 효율을 높였다.

총 3가지 목적을 설정했다



\1. Pushing

\2. Sliding

\3. Pick and Place



**4.2 HER이 성능을 향상 시켰는가?******

 논문에서는 HER이 성능을 향상 시키는지 확인하기 위해서 DDPG를 사용하여 성능을 비교하였다. 

매 transition 마다 replay buffer에 두번씩 저장해줬다. 한번은 에피소드를 만드는데 사용된 goal을, 다른 한번은 final state에 해당하는 goal을저장하였다. 논문 뒤에서 replay 를 위한 goal을 정하는 전략(S)에 대한 ablation study들을 진행했다. 

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA4MDlfMjMg/MDAxNjI4NDcyMDMzOTE3.-vS8ZyptLpMLoD8dAylxRTvs21Qjhfv3wIgbi0VA8pEg.bZtg6xqfivEiTWFEMZx3KYcw5NxqBGge7zVSjbJJxSAg.PNG.nswve/Screenshot_from_2021-08-09_10-20-25.png?type=w966)                                                                    

사진에서 보면 HER이 없는 DDPG는 문제를 풀지 못한다는 걸 알 수 있다. 반대로 DDPG+HER은 거의 완벽하게 풀었다. 



**4.3 HER은 goal이 한개라도 성능을 향상 시켜주나?**

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA4MDlfMjQ2/MDAxNjI4NDczMzY4ODQy.v_2lEQtH9Xr-z6f4ruX3aKWjYls75zfxoiGa1EJxrKEg.jxL1IOxxpOt1z_Fq5n-Z5N52bnhTo-CQGail5xz7Mksg.PNG.nswve/Screenshot_from_2021-08-09_10-42-38.png?type=w966)                                                                    

단순히 goal이 하나라도 HER은 성능을 향상 시켜주는지에 대한 내용이다. 같은 실험이지만 goal을 모든 에피소드에 동일하게 진행하였다.

당연하게도 DDPG+HER이 더 좋은 성능을 보여준다. 신기한점은 HER은 Multi-goal 환경에서 더 빠르게 학습한다는 점이다. 따라서 만약 한개의 goal만 찾으려고 할 경우에도 Multi goal 로 학습시켜주면 더 빠르게 수렴한다. 



**4.4 reward shaping과 HER의 관계는 어떨까?**

논문에서는 이진 형태의 보상만 고려하고 있었다. 

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA4MDlfNjgg/MDAxNjI4NDczOTkxODc2.9P0GLTX8KaebeY1T9JlgURnn2h5k-VJ5W2kCZpJX5lcg.dj1AGhb_jTq6VsJKqrqOEi4Hb1DIlNEl1Aqav90OBoYg.PNG.nswve/Screenshot_from_2021-08-09_10-53-02.png?type=w966)                                                                    

그렇다면 DDPG+HER 에서 reward를 다르게 작성해주면 어떻게 됄까? 보상 함수를 아래와 같이 변경해보자.

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA4MDlfMTM4/MDAxNjI4NDc0MDU2MTQ3.EyKyzqK9PxhMFiGPjCJxHMef618O_UB3HPukeglkDdQg.gN-brvG-3JYFrvQEiyCDzCtpE-JEqrmD54EAzxdKz6Ig.PNG.nswve/Screenshot_from_2021-08-09_10-54-08.png?type=w966)                                                                    

여기서 s'은 action을 한 다음의 env의 state이다. 

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA4MDlfMTU0/MDAxNjI4NDc0MjE2NTE1.sfOlddRgJIoqr-hSuQQOM6HlRhEX1dGz8v868ONljQcg.YcMj4etMlIC0oLAmk2b2aI_0qd6wmJSexuALK2TO_qgg.PNG.nswve/Screenshot_from_2021-08-09_10-56-42.png?type=w966)                                                                    

신기한점은 DDPG, DDPG+HER 모두 보상 함수로 문제를 해결하지 못했다. 이런 결과는 사실 매니퓰레이션에 적용되어 성공적인 강화학습들은  논문에서 시도한 보상함수 보다 더욱 복잡한 보상함수를 가지고 있다는 걸 보여줬다. 



짜여진 보상함수 결과가 별로 좋지 않은 이유에는 2가지가 있다.        

​                 

1.우리가 최적화 하려는 것과 성공 조건이 동떨어져 있다. 

\2. 짜여진 보상들은 마이너스가 되는 행동들에 패널티를 주는데, 이는 exploration 자체를 방해할 수 있다.



이는 로봇 팔이 정밀하게 움직이지 않는 이상 상자를 건들 수도 없게 되는 경우를 만들기도 한다.

이런 결과는 domain-agnostic 보상을 짜는 것은 좋은 결과를 만들기 어렵다는 것을 시사한다. 물론 모든 문제에는 이를 쉽게 해주는 보상이 존재하지만, 그런 보상 함수를 작성하는 것은 도메인 지식을 많이 필요로 한다. 이는 policy를 직접적으로 작성하는 것보다 쉽지 않을 수 있다. 따라서 논문에서 제시하는 것 처럼 이진적이고 희소한 보상으로부터 학습하는게 중요한 이유다.



**4.5 얼마나 많은 goal들을 한 trajectory에서 replay해야되며 어떤걸 선택해야 할까?**

HER에서 goal을 고르는 전략에 대해서 이야기해보려 한다. 



기존에는 env에서의 final state를 goal로 설정했었다. 이를 strategy final이라 불었었다. 이거 외에도 아래 3가지 방법들이 있다.

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA4MDlfMTcy/MDAxNjI4NDc2MzA5MTcz.bhDy2b-EYjxn26RLX1vnVzozSAgMwogX7PI3OxLJvs8g.4RjTwB898TeW6MRiRVUFMfEN6iq68_i9gATErkdvEO0g.PNG.nswve/Screenshot_from_2021-08-09_11-31-41.png?type=w966)                                                                    

모든 전략들이 파라미터 k를 가지고 있는데 이는 일반적인 experience replay와 HER data의 비율을 조절한다. 

각 전략을 비교한 그래프는 아래 사진 참고

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA4MDlfMTgy/MDAxNjI4NDgxNDcyMjk0.zJfkCif75owZo3IRE4sOlIxY6O4wQW514MkVDfl_POEg.ZQ8GCWxcesG8KmSJXxCT0rcLaDsBaQ531dadUALcokwg.PNG.nswve/Screenshot_from_2021-08-09_12-57-41.png?type=w966)                                                                    

그래프에서 알 수 있는 것은 가장 중요한 goal은 가까운 미래에 도달할 수 있는 goal이라는 점이다. 또한 K 값이 8 이상일 때는 성능이 저하되는 것을 볼 수 있는데, 이는 버퍼 속에 있는 일반적인 replay data가 매우 낮아지기 때문이다.



**4.6 Deployment on a physical robot**

논문에서는 시뮬레이션에서 학습 된 pick-and-place task을 바로 가져와서 실제 fetch robot에 튜닝없이 적용했다고 한다. 상자 위치 예측은 CNN을 통해서 진행 됐다. 처음에 policy는 5번중 2번을 성공했지만 box 위치에 대한 estimation에 오류가 있을 경우 강건성을 보장하지 못했다. (시뮬레이션에서는 state가 완벽하게 들어오기 때문) 가우시안 노이즈를 추가하여 재학습시키니 성공률이 5/5였다. 

​                                                                                                                                                                                                                                                                                                                                                                                             

**[Conclusions]**

RL알고리즘을 희소하고 이진 적인 보상을 가지는 문제에 적용할 수 있게 도와주는 HER이라는 방법을 제시한다.

HER은 바닐라 pg가 하지 못한 일을 가능하게 했다. 또한 현실에서 시뮬레이션 한 모델을 튜닝없이 바로 적용할 수 있다.