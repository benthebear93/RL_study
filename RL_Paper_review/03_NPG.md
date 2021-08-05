### [Abstract]

 논문에서는 파라미터 공간의 존재하는 steepest descent direction을 표현하는 natural gradient  방법에 대해서 제안한다. 또한 gradient 방법이 파리미터 값 안에서 큰 변화를 만들수는 없지만, natural  gradient는 단순히 좋은 action이 아니라 greedy한 최적의 action을 선택하는 것을 보인다. 이 greedy한  최적의 action은 서튼이 제안한 compatible value function을 근사하는데 사용 된 policy  iteration의 improvement step 것과 같다. 간단한 MDP와 복잡한 MDP(테트리스)에서 크게 향상 된 결과를  보여준다.

 

### Introduction

큰 MDP에서의 Policy gradient 방법들은 많은 관심을 받았는데, 이 방법들은 미래의 reward의 gradient를  따라가는 제한 된 policy들 중에서 좋은 policy를 구하였다. 하지만 일반적인 gd rule은  non-covariant하다. 

![img](https://blogfiles.pstatic.net/MjAyMTA3MzFfNjYg/MDAxNjI3NjU3Njc5NzAz.YYLinHROtYarl2kVwT8KQvtYVDngg9IAoedH9SNEuM0g.xmD8J11EoESk-byeoBFl5zB6C6tjK0tU9u_k2YP_9fkg.PNG.nswve/1.PNG?type=w1)

사진 설명을 입력하세요.

이 뜻은 위 식의 경우 좌변이 policy parameter인 세타의 단위이고, 우변이 1/세타가 단위이기 때문에  식이 차원적으로 일치하지 않는다. 

(모든 policy parameter가 차원이 같지는 않다.)



이 논문에서는 policy의 내재 된 구조에 기반하여  metric을 정의하여 covariant gradient를 제시한다.  ngd가 greedy한 최적의 action을 선택하는 방향으로 간다는 걸 보인다. 이 방법을 사용하면 plateau 현상 같은  것은 문제가 아니라는 걸 보인다. 



### A Natural Gradient

먼저 모든 policy가 ergoic하게, 즉 잘 정의된 stationary distribution을 가지고 있다고 가정한다. 이때 아래 식들은 만족하게 된다.

![img](https://blogfiles.pstatic.net/MjAyMTA3MzFfMjk1/MDAxNjI3NjU5OTEwODQy.0Knjq__HqCZmOBAuF6SDW597HUTRAhaj2Bx_lwSiOS0g.PhzsO12bDmnpeYlkqwea5-4T7c-9pyL0vSmN1NeT-yUg.PNG.nswve/2.png?type=w1)

사진 설명을 입력하세요.

논문에서는 조금 더 복잡한 상황을 고려한다. 이때 복잡한 상황이란 부드럽게 파라미터화 된 policy들의 제한된 클래스에서의 평균 리워드를 최대로 만드는 policy를 찾는 것에 agent의 목적인 경우를 말한다. 

파라미터화 된 policy는 아래와 같이 표현한다.

![img](https://blogfiles.pstatic.net/MjAyMTA3MzFfMjQg/MDAxNjI3Njk5NTg5NzE2.P_cdcC5ExnRtyGCQMxkCA79yP6lycYXt2KloBitPupMg.5GxvrGjmrj07xPlKs_EAzrPNM0kFoe9-nJxToyddbYcg.PNG.nswve/image.png?type=w1)

사진 설명을 입력하세요.

평균 리워드의 정확한 gradient는 아래와 같다.

![img](https://blogfiles.pstatic.net/MjAyMTA3MzFfMjgy/MDAxNjI3Njk5NjQ0NzI0.ETtWBE0RrEQInnakimXqkHnMzjPZdWOjKqFWeZb24QQg.P7sIdVkTY69OfJfAXR29uwEtyNPNeawe3q-l1H3P7Wgg.PNG.nswve/image.png?type=w1)

사진 설명을 입력하세요.

steepest descent 방향의 평균 리워드는 벡터 dtheta 로 정의 되는데, 이는 dtheta 제곱의 값이 작은 상수로 정의된 상태이다.

이때 dtheta의 값는 평균 리워드에서 dtheta만큼 바뀌는 값을 최소화 하는, 즉 기울기를 최소화하는 값으로 정의 된다(...확실하진 않음)



이때 dtheta의 제곱은 positive defined matrix G에 관해서 정리할 수 있다.

![img](https://blogfiles.pstatic.net/MjAyMTA3MzFfMTIy/MDAxNjI3NzAwMjUxNjY4.2gdNCDCQgQ4V2jpdYL4jmRr4H9wQ5lOKLxsVd2FYtDIg.hsSPLmUC8On4LzagGYjrw99-3mQunUjzo0Qa7TnAGpMg.PNG.nswve/image.png?type=w1)

사진 설명을 입력하세요.

따라서 steepest descent direction의 경우 아래와 같이 나타낼 수 있다.

![img](https://blogfiles.pstatic.net/MjAyMTA3MzFfNzUg/MDAxNjI3NzAwODA0OTU3.WaAAdh0A7jhVrUtgp_3F3RYYnnJ24bOAwyT8e8LIw6Ag.g74gzNLbkdBeo_9XQZGlR0wpIdAGb0rK8GC8aSRSikQg.PNG.nswve/image.png?type=w1)

사진 설명을 입력하세요.

standard gradient descents는 평균 리워드의 그라디언트 방향으로 따라가게 되는데, 이때는 steepest descent인데  positive deifned matrix G가 Identity matrix(I)인 경우이다. 즉 standard gradient descent는 steepest descent의 특수 케이스 라는 것이다. 하지만 metric의 hoc choice의 경우 꼭  알맞은 것은 아니다. 좌표 기준으로서 metric을 정의하는게 아니라, 좌표가 파라미터화 될 수 있는  manifold형태로  정의하는 것이 좋다. 이때 manifold로 정의 된 metric이 natural gradient를 정의한다. 



평균 리워드는 사실 분포의 집합에서 함수이다. 각 state s마다 probability manifold가 있는데, 이때 분포는 좌표 theta를 가지는 manifold 위에 있는 점이 된다. 이 분포의 Fisher information matrix는  Positive definite이며 아래와 같다.

![img](https://blogfiles.pstatic.net/MjAyMTA4MDNfMjU5/MDAxNjI3OTU1NTY4ODQw.l7O3L-rVQARo_Mlw18HMM_ljTikiyW84BhyAC3-LrLsg.zjKpb_NR8HxTACmDC_mbZIPTK85mpQLv4Wq3n7oHaFog.PNG.nswve/Screenshot_from_2021-08-03_10-52-33.png?type=w1)

사진 설명을 입력하세요.

Fisher information matrix는 파라미터 공간 확률분포에서 invariant metric이다. 좌표 선택과 무관하게 두 점의 거리를 정의한다는 관점에서  invariant하다는 점을 알 수 있다. 위 분포에 의해서 평균 리워드가 정의되므로, metric 선택은 아래와 같다. 

![img](https://blogfiles.pstatic.net/MjAyMTA4MDNfMTcz/MDAxNjI3OTU1OTcyNDgx.9mlCiB4Q9ZdCYWB-LmOEqX_Ns07tXDrNQYjiiSEOP5kg.nZ2-WZ9yR4Jd7v9e2CJ1KTw7OiUnKqnKvb2WpeVGoi0g.PNG.nswve/Screenshot_from_2021-08-03_10-59-26.png?type=w1)

사진 설명을 입력하세요.

FIsher information matrix가 간단하게는 state간의 거리를 정의한다고 볼 수 있다. 따라서 각 state들의 거리  평균이 F(theta)가 된다. 이로 인해 steepest descent direction식은 아래와 같아 진다. 

![img](https://blogfiles.pstatic.net/MjAyMTA4MDNfMTYy/MDAxNjI3OTU2NDk5Nzg5.u7ugSSuVDWb2q1YXJA6x2Kh5CV504qGvyWpB-unTzlUg.bitOIMGMRm23MZ18Td1ERAYZ5PgLTOWE795a6G7cym0g.PNG.nswve/Screenshot_from_2021-08-03_11-08-10.png?type=w1)

사진 설명을 입력하세요.

### The Natural Gradient and policy iteration

Natural Gradient를 사용한  policy iteration과 일반적인 policy iteration을 비교해보자. 

비교를 확실하게 하기 위해 action value function Q를 compatible function approximator f(s,a;w)로 근사한다고 보자.



#### 3.1 Compatible Function approximation

벡터 theta, w에 대한 정의는 아래와 같다.

![img](https://blogfiles.pstatic.net/MjAyMTA4MDNfMTAz/MDAxNjI3OTU3ODQxMzg5.6bP7IoDyGCErVmrNWWPEGPQzzMEZ4LUnhxCXNYyzDtYg.6wJE6dvRc98PfdjfUzkYF2w7zMCj7yt4e6nUNDzodSwg.PNG.nswve/Screenshot_from_2021-08-03_11-30-12.png?type=w1)

사진 설명을 입력하세요.

![img](https://blogfiles.pstatic.net/MjAyMTA4MDNfMjc0/MDAxNjI3OTU3ODM3MzU1.1bW3EirELqC1lHyeXzG2W61299fALPdRV0Q2ssVFPEIg.WszlD5ZJ2d3xkN9tE1LpiLDef2ym-dd-y7f9tb7rQE0g.PNG.nswve/Screenshot_from_2021-08-03_11-30-19.png?type=w1)

사진 설명을 입력하세요.



이때 w~가 squared error인 아래 식을 최소화 하게 한다.

![img](https://blogfiles.pstatic.net/MjAyMTA4MDNfMzgg/MDAxNjI3OTU3OTAxMTM2.0hBM2AVMEGd3GZh2NSd56T09ftL6qn8uPVmApvCIaw0g.IT9q-CxO6n2UJPGVSNGWdrMtadOxp5fbuUS1rJKj-A8g.PNG.nswve/Screenshot_from_2021-08-03_11-30-28.png?type=w1)

사진 설명을 입력하세요.

이때 function approximator를 gradient를 계산하기 위해 true value를 대신 사용할 수 있다는 점에서  function approximator가 policy와 compatible하다. 대신 사용하여도 결과는 같다. 



#### Theorem 1.

w~가 squared error e(w,pi)를 최소화 하게 한다. 그러면 아래 식처럼 w~와 steepest descent gradient 와 같아진다.

![img](https://blogfiles.pstatic.net/MjAyMTA4MDNfMjA3/MDAxNjI3OTU4Mzc1NzU4.96ZZGizGhgzNCuCTJkmRLIo10SshD4hjAxH4rWiF3oog.lMvRUT0wbIOGKmLzLOwnt2MDDe4W0M23mCfDPFIFDAog.PNG.nswve/Screenshot_from_2021-08-03_11-39-28.png?type=w1)

사진 설명을 입력하세요.

증명은 생략한다. 이와 같은 이유로 sensible actor-critic framework는 선형 function  approximator의 weight로서 natural gradient를 사용하도록 되었다. 만약 function  approximation이 정확하다면, 좋은 action은 natural gradient와 내적 값이 매우 큰 feature  vector를 가진다.



#### 3.2 Greedy Policy Improvement

 논문에서 언급한 function approximator를 사용하면 state s 에서 선택하는 action a는 높은  value를 가지는 쪽으로 진행된다. 이 단락에서는 natural gradient가 단순히 good action이 아닌 best  action의 방향을 움직인다는 걸 보인다.



exponential family 안에서의 policy를 고려해보자. exponential family를 고려한 이유는 이들이 affine  geometry이기 때문이다. 따라서 tangent vector에서 한점이 변환되어도 manifold 위에 있게 된다. 일반적으로  policy 파라미터와 state에서의 action의 policy의 probability manifold는 곡면 일 수 있기  때문에 tangent vector에 의한 point의 변화는 꼭 manifold 위에 있을 수 있는 것은 아니다.  



#### Theorem 2.

이번엔 exponential family에 대해서 natural gradient에서 충분히 큰 step으로 학습을 설정한다면 greedy policy improvement step 이후에 발견한 policy와 동일함을 보인다. 즉 natural gradient + 충분히 큰 step  = policy after greedy policy improvement step (라는 뜻..)

![img](https://blogfiles.pstatic.net/MjAyMTA4MDRfMTQ3/MDAxNjI4MDY1NDE4MDEz.3IzO9_LkfzmG2xz7jJE94nAbtVJWVkfJl7mD7fdhPT0g.XA9ZY-n0YLS7XPtiqclYfwOjVy3tWO4ybQKmGwicE5cg.PNG.nswve/Screenshot_from_2021-08-04_17-23-25.png?type=w1)

사진 설명을 입력하세요.

policy가 exponential family에 속할 때 평균 리워드 gradient가 0이 아니고, w~가 approximation error를 최소화 한다고 하자.

또한 일반적인 policy 를 policy 파라미터가 평균 reward gradient 방향(w~)으로 무한히 내려가는 값이랑 같다고 할 때 

![img](https://blogfiles.pstatic.net/MjAyMTA4MDRfMjAg/MDAxNjI4MDY2MzE1ODI0.Qw_8qByEtBD3SipNvNr-jQKziYhQwa4FwdK1VyVAe2Ag.S4yvdLitPizYtMlUZ7aFrWyjMiXrUdKUP0lky-uZLxMg.PNG.nswve/Screenshot_from_2021-08-04_17-38-19.png?type=w1)

사진 설명을 입력하세요.

위 식을 만족할 경우에만 policy는 0이 아니다. 

증명은 생략한다. (사실 이해가 잘 안된다.)



위와 같은 내용으로 natural gradient는 best action을 선택하는 방향으로 이동한다는 것을 알 수 있다. 만약  standard non-covariant gradient rule이 단순히 policy 대신 사용됐다면 오직 더 "좋은"  action만 취할 것이다. 즉, function approximator의 Expection보다 큰 값을 가지는 action을  선택하게 된다. exponetial family의 사용은 learning rate이 무한한 경우 같은 극단적인 경우에서를 보이기  위함이었다.



다시 일반적으로 파리미터화 된 policy로 돌아와서 이야기해보자. 아래 나오는 3번째 정의는 natural gradient가  지역적으로 best action을 향해 이동하고, Q를 위한 ocal linear approximator에 의해 움직인다는 것을  보인다.



#### Theorem 3.

w~파라미터가 예측값의 에러를 최소화 해주고, policy 파라미터의 업데이트를 아래와 같이 해준다.

![Screenshot_from_2021-08-05_16-49-08.png](https://blogfiles.pstatic.net/MjAyMTA4MDVfMjcz/MDAxNjI4MTQ5NzY4MjY2.zIcDJz08EA-YDLyuqPGQ6IJlOrphv589PhhPw4NDTvUg.50LaZ12GK_Qg-i8rZUz6Kd5zVryp_nEI69GdokkFrjkg.PNG.nswve/Screenshot_from_2021-08-05_16-49-08.png?type=w1)

사진 설명을 입력하세요.

그러면 업데이트 된 policy의 경우 아래와 같이 바뀐다

![img](https://blogfiles.pstatic.net/MjAyMTA4MDVfMjQw/MDAxNjI4MTQ5NjUxNzk2.3up9BVR0idCpRdvUssYbFuggOe5A8FKbpN6sMgphJY8g.bhMQEpC3edKoO2cpA_uy7Yjb_KvZAbDOaE81aSxL_MUg.PNG.nswve/Screenshot_from_2021-08-05_16-47-18.png?type=w1)

사진 설명을 입력하세요.

증명, 테일러 급수 1차로 근사하면 아래 식과 같이 된다. 그 외에는 증명1,2를 참고한다.

![Screenshot_from_2021-08-05_16-53-18.png](https://blogfiles.pstatic.net/MjAyMTA4MDVfMTc3/MDAxNjI4MTUwMDA2NjIx.h82pngsqRDI1agO1FGbbQsqPHsEqc7JaILSyYO_JLJYg.h4RVCayyQI6sIkQhQsu7AewIpHQZp8jmC2MWe1HYXF0g.PNG.nswve/Screenshot_from_2021-08-05_16-53-18.png?type=w1)

사진 설명을 입력하세요.

흥미로운 점은, greedy action을 선택한다고 해서 일반적으로 policy를 향상시키지는 않지만 ,

 natural gradient를 통 할 경우 향상을 보장할 수 있다고 한다. 초기 향상 또한 F가 positive definite이기 때문에 보장된다.



**[Metrics and Curvatures]**

Fisher information matrix인 F의 선택은 하나가 아니기 때문에 F보다 좋은 metric이 있는지에 대한 의문은 당연하다. 

다른 파라미터 추정의 세팅에서는 FIsher information Hessian에 수렴한다. 따라서 이는 asymtotically  efficient하다. 즉 Cramer-Rao bound를 얻을 수 있다. 단 논문의 경우는 blind source  separation 경우와 더 가깝다. 이 경우 metric이 파라미터 공간에서 근하여 정해지고 asymtotically  efficient하지도 않다. 즉 second order convergence 를 보장하지 않는다. blind source  separation 경우를 위한 방법으로는 Marckay 는 Hessian의 data independent term에서  metric을 빼야 한다고 주장했다. 



이전 섹션에서 논문에서의 선택이 맞다는 것을 보였지만 Hessian과  F가 어떻게 관련되어 있는지를 알아보려 한다. 

![Screenshot_from_2021-08-05_19-04-41.png](https://blogfiles.pstatic.net/MjAyMTA4MDVfMjA3/MDAxNjI4MTU3ODkxNTI4.731CiyQNdT663Hem6aEXz6Uno4BfptfUJBvhAx2vB-4g.Cbo6eglxcvhmw9hDFztUpG1qb2sZWQTflSSZfvUdKKsg.PNG.nswve/Screenshot_from_2021-08-05_19-04-41.png?type=w1)

사진 설명을 입력하세요.

average reward 값을 policy 파라미터로 두번 미분한 식이다. Hessian에서 모든 항들은 data dependent한데,  (이는 state와 action value 값과 엮여있다는 뜻이다.) F가 마지막 2개의 항에서 어떤 정보도 가져오지 않는다는  것은 명확한데, 이는 policy를 2번 미분한 항 때문이다. 하지만  Q값이 policy의 curvature에 가중을 주면서  metric에서는 가중을 무시하게 된다.  (무슨 뜻이지..?)



blind source separation 경우와 비슷하게, 논문에서 이야기하는 metric은 Hessian에 꼭 수렴할 필요는 없고  asymtotically efficient할 필요도 없다. 하지만 일반적으로 Hessian은 positive definite 하지 않을 것이고, curvature도 policy 파라미터가 local maxima에 도달하기 전까지는 잘 사용될 수 없을 것이다. local maximum 근처에서는 conjugate method를 사용하는 게 더 효율적일 것 이다.



**[experiments]**

 중략



**[Discussion]**

 gradien 방법이 greedy policy iteration 방식에 비해 policy를 크게 바꿀 수는 없지만 섹션3이  언급하는 것 처럼(natural gradient 방법이 policy improvement 방향으로 나아간다는 점에서) 두 방법을  분리해서 볼 것은 아닌 것 같다. line search의 overhead(?)와 같이 하면 2 방법은 더욱 비슷해진다.  장점이라하면, greedy policy iteration step에서와는 다르게 선능 향상이 보장되었다는 점이다.



아쉽게도 F가 Hessian에 asymptotically하게 수렴하는 것은 아니며 따라서 conjugated gradient method가 더욱 asymptotical 하지만

수렴 포인트로부터 먼 경우 Hessian이 꼭 중요하게 필요한 것은 아니다. Natural gradient가 더욱 효율적일 수  있다(Tetris experiment를 참고하면 알 수 있다). 직관적으로 생각했을 때 natural gradient가  maximum에서 멀 경우 더 좋은 이유는, greedy optimal action들을 선택하는 방향으로 policy가 진행되기  때문이다. 종종  maximum(수렴점)에서 먼 곳에서 performance 변화가 크게 일어나는 지역이기 때문이다. 비록  conjugate 방법이 maximum 지점에서 빠르게 수렴할지라도 따라오는 performance 변화는 무시될만 할 것이다. 