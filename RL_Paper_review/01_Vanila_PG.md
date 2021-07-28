\#강화학습 #논문리뷰 

**Sutton PG**

**논문 제목  : Policy gradient methods for reinforcement learning with function approximation Policy Gradient Algorithms**

**논문링크 :** **https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf**

**https://github.com/benthebear93/RL_practice** **(추후 공개예정)**



**[Introduction****]**

최근 10년간 RL에서 주로 사용 된 방식은 value function approach 였다. 즉 모든 function approximation은 value function을 예측하는 데 사용됐다. 값을 예측하는데 action-selection 정책으로 greedy 방법이 암묵적으로 사용됐다. 이들은 한계점이 있었는데



\1. 최적 정책들은 주로 stocastic(확률론 적)으로 특정 확률에 따른 action을 취하는 반면, deterministic policy를 방향으로 진행됐다.

\2. 액션에 대한 value값의 작은 변화도 액션이 선택 되고 안되고를 결정할 수 있다. 이런 불연속적인 변화는 value  function 방식의 알고리즘을 사용하는데 있어서 수렴을 보이는데 문제가 된다. 예를 들어서 Q러닝, Sarsa, dp 방법들은 간단한 MDP들이나 function approximation에 대해 어떤 정책으로도 수렴하지 못했다. 이는 정책을 바꾸기 전에 각 스탭마다 최적의 근사를 찾거나 mse나 residual gradient, td, dp 방법에 대한 best인 경우에도 수렴하지  못하고는 한다.



이 논문에서는 RL에 사용되는 function approximation에 대한 대안을 제시한다. value function을  근사하여 deterministic policy를 계산하는 것이 아닌, 독립적인 function approximation를 사용하여 stochastic policy를 직접적으로 근사한다. 



예를 들어 정책은 input이 state, output이 action selection 확률, weight가 policy  parameter인 신경만으로 표현될 수 있다. 정책 파라미터의 벡터로 세타를 두고 step에 대한 평균 reward를  고려해보면, policy 파라미터는 gradient에 비례하도록 업데이트 된다. 

![img](https://blogfiles.pstatic.net/MjAyMTA3MjNfMjQy/MDAxNjI3MDE3NTU3MjI2.J7kYWOhjo6NaPt4FzEsAnSSx0ORqwNvE1_SzJHRsKOgg.0Ir9aNjRwMfdvii1nynqSXDw9LAI9EPKwWMFaa0qXOIg.PNG.nswve/Screenshot_from_2021-07-23_14-19-03.png?type=w1)

사진 설명을 입력하세요.

이때 기호 알파는 step size이다. 만약 위와 같은 근사가 이뤄질 수 있다면 policy 파라미터인 세타는 performance measure에서 local optimal policy에 수렴하도록 보장 된다. value function 방법과 다르게 세타에 대한 작은 변화는 오직 policy의 작은 변화만 만든다. 



**[****Policy Gradient Theorem]**

MDP를 따르는 RL환경이라고 본다.

function approximation으로 agent의 목적을 수식화 하는데는 2가지 좋은 방법이 있다. 첫번째는 스텝에 따른 reward의 기대값에 대해서 정책들이 rank되어 있는 형태의 평균 reward 수식이며, 

![img](https://blogfiles.pstatic.net/MjAyMTA3MjNfMjcg/MDAxNjI3MDI3Mzc2ODg2.KSpujtGT28J0DGZDW1caA1IvFhTy70sYEf2YKTedBDYg.Dn-SmifYem07s9LZxMtlfdzd5ObSevqVTjlbkSiKTe8g.PNG.nswve/Screenshot_from_2021-07-23_17-02-43.png?type=w1)

사진 설명을 입력하세요.

두번째 식은 특정 시작 state가 있을 때 정책에 따른 장기간 얻어진 reward에 대한 기대값이다. 

![img](https://blogfiles.pstatic.net/MjAyMTA3MjNfMjIx/MDAxNjI3MDI3Mzk0OTQx.sgoWRt6ewJlE95qOhr2tN0wK_IOjHQMFTQYgvQCxPZQg.0J5BYQthvDZRq7Finl7faJTbPAAocaToGz6YySeqwqAg.PNG.nswve/Screenshot_from_2021-07-23_17-03-01.png?type=w1)

사진 설명을 입력하세요.

**[****Theorem1]**

![img](https://blogfiles.pstatic.net/MjAyMTA3MjNfMjQw/MDAxNjI3MDI3NzcyNTMw.fqPkZXu79XWKYZKjCXTvTLPrwcA6tgjPhd_0-aRCdHUg.GZpmq9qRQiePoePiWT_dQlKsKBLKbMdKwGHr-kHHsikg.PNG.nswve/Screenshot_from_2021-07-23_17-09-21.png?type=w1)

사진 설명을 입력하세요.

위와 같이 gradient를 표현하는 것은 average-reward 방식의 value function approximation과  논문에서 제시하는 start-state formulation 두가지 방법 모두에 적용가능하다. 여기서 gradient를 이렇게  표현하면 좋은 점은 policy가 변함으로서 state의 분포가 영향을 받지 않는 다는 점이다. 이는 sampling 기반으로  gradient를 구하는데 편리하다. 예를 들어서 만약 특정 policy를 따르는 분포에서 state s가 샘플 됐을 때, 

![img](https://blogfiles.pstatic.net/MjAyMTA3MjNfMTY4/MDAxNjI3MDI4MDUyMDY0.hJPVVnOMAbYEn2nEMLqHd_JsqmJQoYJQG09EAc98Hqsg.ow5VdOIZPZ523OCti1W3bCFCFSwLKq_Psf4WFu4nb5Ig.PNG.nswve/Screenshot_from_2021-07-23_17-14-03.png?type=w1)

사진 설명을 입력하세요.

는 gradient의 unbiased sample이 될 것이다. 물론 Q도 estimate 되야하긴 하겠지만, 이는 실제  Return 값을 가져와 사용하면 된다. start-state 식을 이용해서 각 time step의 Q값을 근사하여 볼 수 있다. 이는 William의 REINFOCRE 알고리즘으로 연결된다. william의 REINFORCE 알고리즘에서는 기대값에서  drho/dtheta를 따른다.(?)



**2. Policy Gradient with approximation**



Q 함수가 학습 된 function approximatior에 의해 잘 근사됐다고 해보자. 그러면 정의1번 식을 이용해서 Q함수를  대체하려고 할 수 있다. 파라미터 w와 함께 근사함수 f가 Q 함수의 근사값이 된다고 했을 때, 파라미터 w가 근사함수 f에  대해서 policy를 따라 업데이트 되는 건 자연스럽다. 이때 파라미터 w의 변화량은 근사 된 Q값과 근사함수 f의 차이를  파라미터로 편미분 한 값과 비례한다. 

![img](https://blogfiles.pstatic.net/MjAyMTA3MjhfMjM4/MDAxNjI3NDM4MDI1NzI3._7dTHOMOEb00Q6oCL-V7pG5lNblX0w8LhtA7tPJ07xYg.7yFqz7XfC97wMHYR5ioO_IT4cGIlapHaEHm9sqdnHqAg.PNG.nswve/Screenshot_from_2021-07-28_11-02-17.png?type=w1)

사진 설명을 입력하세요.

이때 예측 Q값의 경우 일반 Q값의 unbiased estimator이다. 이 식들이 다 만족할 경우 local optimum에 수렴하게 된다. 그러면 아래 식을 만족하게 된다.

![img](https://blogfiles.pstatic.net/MjAyMTA3MjhfMTI5/MDAxNjI3NDM4MDE3OTY1.yICRrScpY83pBs-GyBiwjljsthJjX3EBP0Y2OsXSB0cg.BHazTPpVNrtibc5vcdI5bCYgfYiMhP-HRE6xLh4PMGUg.PNG.nswve/Screenshot_from_2021-07-28_11-06-48.png?type=w1)

사진 설명을 입력하세요.

**[****Theorem2]**

 근사함수 f가 위 식을 만족하게 되면 이를 이용해서 정의2는 쉽게 증명할 수 있다.

![Screenshot_from_2021-07-28_11-14-01.png](https://blogfiles.pstatic.net/MjAyMTA3MjhfMTg2/MDAxNjI3NDM4NDQ4ODk1.B6v_zc7v8xG3I2pL4kwwkUD2NOLJVknFg1TvLGSuZLQg.2PlMAh0jjfsBRBkjfUh4ozyPXzNxnlevgF3epn9-7mMg.PNG.nswve/Screenshot_from_2021-07-28_11-14-01.png?type=w1)

사진 설명을 입력하세요.

![Screenshot_from_2021-07-28_11-14-16.png](https://blogfiles.pstatic.net/MjAyMTA3MjhfMjU5/MDAxNjI3NDM4NDcxMzA0.SGyde59nI1jAkNulO_LKSuQ0PsUbq5jYVTd39S9Dyxcg.VqRDGbYyvLAuI2I44BK8bMYNM-OKWJ8Q2RNSiwiuJLQg.PNG.nswve/Screenshot_from_2021-07-28_11-14-16.png?type=w1)

사진 설명을 입력하세요.

위 2개의 식이 정리 된다면, 앞서 나온 식들을 합쳐서 정리하면 아래의 식이 된다.

![Screenshot_from_2021-07-28_11-22-53.png](https://blogfiles.pstatic.net/MjAyMTA3MjhfMjk5/MDAxNjI3NDM4OTgxNjMz.gmXnPapgcW46x__X-L30MtR4bRwbRqzO6DftIvyKo50g.UNCtWQnrNdut8qLerOLDbU0SOIRRCTNvFLFTx1Sempkg.PNG.nswve/Screenshot_from_2021-07-28_11-22-53.png?type=w1)

사진 설명을 입력하세요.

이 식은 근사함수 f에서의 오차가 policy parameterization의 gradient와 직교한다는 것을 보여준다. 식 자체 값이 0이기 때문에 위에서 이야기 했던 policy gradient 정의에서 뺄 수 있다. 따라서 policy  gradient식은 아래와 같이 정리 된다.

![Screenshot_from_2021-07-28_11-25-23.png](https://blogfiles.pstatic.net/MjAyMTA3MjhfMTMx/MDAxNjI3NDM5MTMwOTQy.Vc3kMvdb60SJ11SEH671lTUwy4ynwPR8CEfWCd0sH0Ug.qH5zS0wh2eDnVUzhxfgnrONaZv8YhA50DMXOfUrxI9Ag.PNG.nswve/Screenshot_from_2021-07-28_11-25-23.png?type=w1)

사진 설명을 입력하세요.

사실 정확한 수식에 대한 이해를 한 것은 아니지만, 정의2번에서 말하고자 하는 것은 결국 Q 함수를 사용하지 않고 approximation function으로만으로 policy gradient를 구할 수 있다는 것이다. 



**3. Application to Deriving Algorithms and advantags**



policy parameterization으로 정의 2번은 value function parameterization을 적절한 형태로 유도할 수 있게 된다.

예를 들어서 특징들의 선형 조합으로 Gibbs distribution 형태로 policy가 구성되어 있다고 해보자.

![Screenshot_from_2021-07-28_11-31-52.png](https://blogfiles.pstatic.net/MjAyMTA3MjhfMTAz/MDAxNjI3NDM5NTE5NjE3.fNtf_PeBkCiKLfbu7Rgzzll31VEIKPxkbCkPUMlYY2sg.6lB4To2rKTjcd0JT6QQae5pYqYGyTguLf3dPL1c8Dw8g.PNG.nswve/Screenshot_from_2021-07-28_11-31-52.png?type=w1)

사진 설명을 입력하세요.

프사이는 L 차원의 특징 벡터로 state-action을 나타낸다. 정의2에서 사용했던 내용을 다시 참고하여 아래와 같은 식을 만들 수 있다.

![Screenshot_from_2021-07-28_11-33-44.png](https://blogfiles.pstatic.net/MjAyMTA3MjhfMTI5/MDAxNjI3NDM5NjMxNzMy.rw90o940-My_QbGKWrgK86uEmJmPsEgMdXDRBSykSHMg.jeAKCUzvO-2sb7PVnPLeXRmHlWYlGB-HIxDoAI3na_sg.PNG.nswve/Screenshot_from_2021-07-28_11-33-44.png?type=w1)

사진 설명을 입력하세요.

위 식을 적분하면 근사함수의 natural parameterization을 구할 수 있다. 

![Screenshot_from_2021-07-28_11-36-22.png](https://blogfiles.pstatic.net/MjAyMTA3MjhfODQg/MDAxNjI3NDM5Nzg5ODY1.jGPwcOoei5qVdWLp7tQiXv7pxnPs23yXVR5ALCqfAFkg.y60D4UlnWXSxMmZqQIIpynXgcnqOa7dNwjXtwh8wN0wg.PNG.nswve/Screenshot_from_2021-07-28_11-36-22.png?type=w1)

사진 설명을 입력하세요.

모든 state들에 대해서 zero mean으로 정규화 되어 있지 않다면 근사함수 f는 policy처럼 같은  특징(features)들에 대해 선형이어야 한다. 식을 보게되면 policy*근사함수의 합은 0이 되어야한다는 걸 알 수 있는데, 이때는 근사함수 f를 Q가 아닌 advantage function(A = Q-V)의 근사로 보는 게 더 낫다. 수렴 조건으로  근사함수는 상태와 상태 간의 variation이 존재하지 않도록하고 state에 대한 action들의 상대값을 가지는 것으로  본다. 사실 위에서 나온 식들은 state에 대한 arbitrary function을 value function에 더한 것을  추가하여(advantage 형태로) 일반화 시킬 수 있다. 예를 들어

![Screenshot_from_2021-07-28_11-50-25.png](https://blogfiles.pstatic.net/MjAyMTA3MjhfMTcg/MDAxNjI3NDQwNjM4ODAx.8vOJ-V7VSPGFOihi9kmpzOD9R0-D1XRs0axME_D4PAwg.x0ertf45B1u9HIv-haYafU3fk5DkWEhaFZvHn7n5ROUg.PNG.nswve/Screenshot_from_2021-07-28_11-50-25.png?type=w1)

사진 설명을 입력하세요.

이런식으로 일반화 시킬 수 있다.



논문에서 baseline으로서 arbitrary function은 가치함수 V가 사용되고 policy와 근사함수 f의 변화에는 영향을 주지 않는다고 한다.



**4. Convergence of Policy Iteration with Funtion approximation**



정의2에서 function approximation이 합쳐진 policy iteration의 형태가 locally optimal한 policy로 수렴하는 것을 처음으로 증명했다. 



[Theorem 3] - Policy iteration with Function approximation

policy와 근사함수 f가 compatibility condition을 만족하고, 미분가능한 function approximator이며 아래 식과 같은 방법으로 parameter들을 업데이트 한다면 

![Screenshot_from_2021-07-28_12-49-18.png](https://blogfiles.pstatic.net/MjAyMTA3MjhfMjYy/MDAxNjI3NDQ0MTY3NjUz.VSlSwFx-sPl_AWy-5u4QfPTYSSyWEoQ8fb1R64RrZHgg.aelWNCCQLzYnZtOlL9u38hUA-M5GAEKld4SwgYrhNmIg.PNG.nswve/Screenshot_from_2021-07-28_12-49-18.png?type=w1)

사진 설명을 입력하세요.

bounded reward를 가지는 어떤 MDP라도 policy gradient 가 0으로 수렴한다. 