**DDPG**

**논문 제목  : CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING**

**논문링크 :** [**https://arxiv.org/pdf/1509.02971.pdf**](https://arxiv.org/pdf/1509.02971.pdf)

[**https://github.com/benthebear93/RL_practice**](https://github.com/benthebear93/RL_practice) **(추후 공개예정)**

****

**[Introduction****]**

****

AI의 목적은 가공되지 않은 고차원의 센서 데이터로부터 복잡한 문제를 해결하는 것이다. 이를 위해 DQN 알고리즘이 나오게 됐다. 

DQN에서는 action value function을 예측하기 위해 deep neural network function approximator를 사용하게 된다. 하지만 DQN은 고차원의 observation space 문제를 해결하는데 어려움이 있고, 오직 discrete 저차원의 action space에서만 적용된다는 한계가 있다. 일반적으로 실제 환경에서의 제어 문제는 continous하고 고차원의 action space를 가지므로 DQN을 바로 적용하지는 못한다.



action space를 단순히 이산화 시켜버리면 될 것도 같지만 한계가 존재한다. 그 한계중 하나가 curse of dimensionality(차원의 저주)때문에 생긴다. 이를 해결하기 위해서 이 논문에서는 고차원, continous action space에서 deep function approximator를 이용한  model-free, off-policy actor-critic algorithm 방법을 제안한다. 



기본적인 개념은 DPG인데 large, non-linear function approximator는 복잡하고 불안정하다고 한다. 따라서 DQN에서 적용된 2가지의 방법을 적용하여 발전시켰다고 한다. 



\1. 샘플간의 상호관계를 줄이기 위해 replay buffer로부터 off-policy를 학습시키는 방법

\2. temporal difference backups 중에서 일관적인 target을 주기 위해 target Q network로 학습되는 방법. 

이 두 방법과 batch normalization을 포함해서 DPG를 발전시켜 Deep DPG (DDPG)를 선보인다.



**[Background]**

****

 논문에서는 일반적인 강화학습 환경인 에이전트가 환경과 discrete timestep으로 상호작용하는 하는 구성을 사용한다. 에이전트의 행동은 policy에 의해 정의 되고, action에 대해 state를 probability distribution으로 매핑하는 방식이다. 환경은 stochastic하고 MDP로 모델링 했다고 한다. 



 action value function은 많은 강화학습에서 사용되는데 이는 policy를 따라 action을 취하고 state에 도착했을 때 얻게 되는 모든 return의 기대값을 말한다.

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA1MjhfMjAw/MDAxNjIyMTgyNTE5NzMz.Qeg7-MDrrR-Xrej8O6ijV5-BfhNY1EBlpQgYRD_28QAg.AMOmw6hTUYjfkrUwuwWwzzOX505Cx6MECN0nuGeFYssg.PNG.nswve/image.png?type=w966)                                                                    

많은 강화학습에서는 벨만 방정식이라는 재귀함수 형식을 사용하는데,

(벨만 방정식은 현재 상태의 가치함수와 다음 상태의 가치함수 사이 관계를 식으로 나타낸 것이다.)

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA1MjhfMjY5/MDAxNjIyMTgyNTgxODgx.kd2i2Yvlhk12ZPvMA1n-G54SBPuhOakoxKr_Yg5sDfAg.PkWZsq-0eLwJPaql4onAWvs3MwCoS9PMScRXMm-Lerwg.PNG.nswve/image.png?type=w966)                                                                    

 만약 target policy가 deterministic, 즉 예측가능하다면 벨만 방정식 내부의 기대값을 없앤 형태로 표현할 수 있다. 

(일반적으로 deterministic하다는 것은 어떤 값이 나오도록 예측할 수 있다라는 뜻이다. 따라서 기대값을 찾는 것이 의미가 없어진다)

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA1MjhfMjMy/MDAxNjIyMTgyNzIwNTU4.LzsLD2i07wL08mOz8IM-Nh8-w7RuM9rWfpWHJXMFeEgg.1ez4a9p-XOyPS-pKIGZs9KXOFtWU9fW5k7LLUfi7cowg.PNG.nswve/image.png?type=w966)                                                                    

이렇게 기대값은 Environment인 state와 reward에만 영향을 받게 되고, 이로 인해 또 다른 stochastic behavior policy로 생성 된 sample을 이용해서 off policy 형태로 학습 시킬 수 있다는 걸 뜻한다.

Q-learning이 일반적으로 사용되는 off-policy 방법이다. Q-learning은 greedy policy를 사용하는데 우리는 function approximators를 파라미터화 하여 생각해 볼 수 있다. 또한 이 파라미터는 loss function을 최소화 하여 최적화 시킬 수 있다.

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA1MjhfMjM1/MDAxNjIyMTg4NDg2Nzk0.T1a7M1-S74h2prKcR5gAvaW1cHc-qrbdTQqWaPlBK68g.CMZKktdUqdnSGPTJL8Q3tcQXTp2xQtZN95TW3il2V5Qg.PNG.nswve/image.png?type=w966)                                                                    

크고, 비선형인 function approximator를 가치함수(value function)나 action-value_function을 학습시키는데 사용하는 것은 지양되었다. 실제로 불안정하고 보장할 수 없기 때문이었는데, 최근 replay buffer와 yt를 계산하기 위한 target network를 분리하는 방법을 사용한 DQN이 large neural network를 function approximator로 사용하여 좋은 성능을 보여줬다. 앞서 말한 것 처럼 이 논문은 이 두가지 방법을 적용하여 만든 DDPG에 대해 이야기 하고 그에 대한 자세한 내용은 Algorithm 섹션에서 이야기하려 한다.



**[Algorithm]**

****

continuous한 action space에서 Q-learning을 직접적으로 적용시키는건 greedy policy를 찾는데 action의 최적화가 매 timestep마다 필요하기 때문에 불가능하다. 따라서 논문에서는 actor-critic 방법을 DPG 알고리즘에 적용하였다. 



DPG알고리즘은 파라미터화 된 actor 함수를 유지하는데 이는 현재 policy를 특정 action에 대한 state를 deterministic하게 mapping함으로서 정의한다. 즉 특정 행동에 대해서 deterministic하게 state를 부여해줘서(mapping) 현재 policy를 정의하는데 이는 파라미터화 된 actor 함수를 유지하게 해준다 라는 뜻이다. 이때 Critic은 Q-learning에서 처럼 벨만 방정식을 통해 학습한다. actor는 policy's performance의 gradient 방향으로 업데이트 되었다.



​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA1MzBfMTYx/MDAxNjIyMzgxNTIxNzI1.A9WhtLuzw2b9nvc93HVZ5BBrfcST_gJ7hFAtFJMbNjQg.Dsw_eSBcJcyfq5PDxxtASPULRI-M1mgpZOMo4yDqBfQg.PNG.nswve/image.png?type=w966)                                                                    

Q-learning에서처럼 비선형 function approximator를 사용하는 것은 수렴성을 보장 할 수 없다는 뜻인데, state space가 클 경우 이런 approximator의 사용은 꼭 필요했다. 하지만 NFQCA 에서는 DPG와 같은 방법으로 업데이트를 하지만 neural network function approximator를 통해 batch learning으로 안정성을 챙겼지만 large network에서는 사용하기가 힘들었다. NFQCA는 매 업데이트 마다 policy를 초기화 하지 않았고 이는 large network와 크기를 비슷하게 맞춰야 했으며, 결국 DPG와 같았다.



 이 논문에서의 기여한 점은 DQN의 성공적인 부분인 neural network function approximator를 large state 와 action space를 online 학습하는데 사용하도록 한 점을 가져와  DPG의 변형으로 만든 것이며, 이를 Deep DPG 라고 한다.



**Replay buffer**

neural network를 강화학습에서 사용할 때의 어려운점은 샘플들이 독립적이로 동일하게 분포되어 있다고 가정하는 것이다. 당연하게도 환경을 순차적으로 탐색하면서 샘플들이 만들어지면 이전의 가정들은 맞지 않는다. 추가적으로 하드웨어의 최적화를 효율적으로 사용하기 위해서는(아마도 컴퓨터 사용의 최적화를 말하는 것 같다) 온라인보다는 mini-batch로 학습하는게 필수다.



가정이 맞지않고 하드웨어를 효율적으로 사용하기 위해 사용하는 방법은 replay buffer이다. 이는 일정한 크기의 캐쉬 형태인데, 탐색 policy를 통해 샘플들이 환경으로 부터 tuple 형태(굳이 튜플형태라고 언급한 이유는 뭘까?) 로 buffer에 저장된다. 꽉차게 되면 오래된 것 부터 삭제시킨다. 매 timestep마다 actor와 critic은 buffer로부터 일정하게 minibatch를 가져와 update한다.

DDPG는 off-policy이기 때문에 replay buffer가 커도 무방하다. 강화학습에서는 correlated 된 샘플들만 가져오는 걸 지양하려고 하는데, replay buffer가 존재하고 크다면 uncorrelate한 샘플 들을 더 가져올 수 있게 된다.



**Soft target network**

직접적으로 가중치들을 복사하는게 아니라, actor-critic을 위해 바뀐 soft traget update 방법을 사용한다. actor-critic 네트워크의 복사본을 만들어서 traget value를 계산하는데 사용된다. 이때 traget network의 가중치는 학습된 네트워크를 천천히 따라서 업데이트 하는 방식으로 업데이트 된다. 

$\Theta "\gets \tau \Theta +\left(1-\tau \right)\Theta "\ with\ \tau \ll 1$Θ′←τΘ+(1−τ)Θ′ with τ≪1

단순히 속도를 늦추는 것만으로 학습의 안정성을 높인다. (강화학습에서 학습의 안정성이란 뭘 말하는 걸까? 수렴한다는 거? 어떤 목표에 수렴한다는 건지 살짝 애매모호하다, 목적을 실패하지 않고 성공할 확률을 말하는 걸까? 속도가 얼마나 느려질까?)

물론 value값에 대한 예측값의 propagation을 traget network가 딜레이 시키기 때문에 느린 학습이지만, 실전에서는 안정성을 위해서 매우 중요한 요소임을 발견했다. 



**batch normalization**

저 차원의 특정 벡터 관찰(low dimesional feature vector observations)로 부터 학습을 할 때 관찰한 것들이 다른 물리 단위를 가지고 있을 수 있다. 마치 속도가 m/s, 위치가 m인 것 처럼. 또한 범위도 환경에 따라 다 다를 것이다. 이렇게 scale들이 다르면 모든 환경에 맞는 하이퍼파라미터를 찾기 어렵고, 효율적인 학습이 어려워진다. 



이런 문제를 해결하기 위한 방법으로는 특징들의 크기(scale)을 조절하는 것이다. 이를 위한 방법으로 batch normalization을 사용한다. 

이 방법은 minibatch에 있는 각 차원의 샘플들을 unit mean과 variance를 갖도록 하는 것이다. 이는 deep network에서 각 차원이 whitened input(필터에서의 개념처럼)이 들어가도록 하면서 covariance shift를 최소화하는데 사용된다.  이를 통해 다양한 단위와 목적을 set range에 들어오는지 확인할 필요없이 효율적으로 학습시킬 수 있었다.



**Noise processor** 

continuous action space에서의 학습의 가장 어려운 부분은 탐색(exploration)이다. DDPG 같은 off-policy의 장점은 탐색 문제 자체를 학습 알고리즘과 독립적으로 둘 수 있다는 점이다. 논문에서는 탐색 policy를 noise process N에서 샘플된 noise를 actor policy에 추가하여 만든다. N은 환경에 따라 선택된다. 논문에서는 Ornstein-Uhlenbeck process를 사용했는데, 이는 temporally correlate noise를 발생시켜 Interia와 관련 있는 물리적 컨트롤 문제에서의 탐색 효율을 높였다. 



**[Results]**

​                                                                            ![img](https://postfiles.pstatic.net/MjAyMTA1MzFfNTYg/MDAxNjIyMzg3Njk1NTU5.kDf_GBjv92nGAF2uByrYan1uleUiIA3jmc9OyyRSvoEg.pSlzaxTHnDcHEGj8AnjLF384GAdmjhH-wmkMaUUoIikg.PNG.nswve/image.png?type=w966)                                                                    



**[Conclusion]**



딥러닝과 강화학습에서의 인사이트를 합치는 결과를 만들었고, 이는 다양한 도메인의 continuous action space에서의 문제를 강건하게 풀 수 있게 했다. DQN의 Atari domain 솔루션보다 DDPG가 짧은 시간내에 수렴점을 찾았으며 충분한 시간이 있다면 더 복잡한 문제도 풀 수 있을거라 생각한다고 한다. DDPG로 한계점이 존재하는데, 대부분의 model-free RL 방식과 마찬가지로 학습시간이 많이 필요하다. 