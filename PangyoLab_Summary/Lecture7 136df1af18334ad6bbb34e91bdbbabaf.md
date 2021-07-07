# Lecture7

### Policy Gradient

### Policy based RL

Lecture6에서는 파라미터 w 혹은 theta를 가지고 state-value function과 action-value function을 근사하는 방법에 대해서 공부했다. 이때 MDP를 모르기 때문에 MC나 TD를 이용해서 evaluation을 하고 입실론 그리디를 통해 improvement를 했다. 이때 우리는 value 함수만 존재했지 policy는 없었는데, 이번 lecture를 통해서는 policy를 특정 파라미터들로 표현하는 것에 공부하겠다. 

### Value-Based and Policy-Based RL

강화학습을 조금 분류하면 3가지로 나눌 수 있다.

![Lecture7%20136df1af18334ad6bbb34e91bdbbabaf/Untitled.png](Lecture7%20136df1af18334ad6bbb34e91bdbbabaf/Untitled.png)

1. Value Based
    - linear value function
    - lmplicit policy(입실론 그리디)
2. Policy Based
    - No value Function
    - Learnt Policy
3. Actor-Critic 
    - Learnt Value Function
    - Learnt Policy

1번 value based는 value function을 만들어서 그 함수가 정확한 값을 return하도록 학습하는 방법이다. 이 방법의 policy는 implicit policy이다. 

2번 policy based는 value function을 학습하지 않고 바로 policy를 배우는 것이다. 

3번은 value function과 policy 둘 다를 학습하는 방법으로 Actor Critic 방법이 있다.

이번 Lecture7에서는 2번과 3번에 대해 이야기해보려 한다. 그 전에 왜 value function을 사용하는 방법에 대해 쭉 배웠는데 policy based를 또 배워야할까? 장점이 뭘까?

### Advantages of Policy Based RL

이점은 수렴성이 좋다. 수렴이 잘 안되면 학습하는데 결과를 얻기가 힘들다. 만약 action의 갯수가 많거나 continuous 한 action 문제일 경우 단순히 최고의 action을 찾는 것만으로도 또 하나의 최적화 문제가 될 수 있다. stochastic policy를 학습할 수 있다. 여기서 stochastic이란 확률적이라는 뜻으로, 기존에는 deterministic한 policy 였다.

단점은 local minimum에 빠질 수가 있고 policy를 학습하는 방법이 variance가 크고 비효율적일 수 있다. 

주로 Policy 기반을 사용하는 이유는 강의를 들어보니 '단순 deterministic하면 문제를 풀 수 없는 경우, 예를 들어 누군가과 가위바위보를 하는 경우에 deterministic하게 내가 낼 가위,바위,보가 정해져 있다면 문제를 풀 수 없다. 그리고 MDP를?MP를 Fully observable 하지 않을 때, 즉 환경에 대해 완벽히 알지 못해서 내가 만드는 feature에 구멍이 있을 때' 인 것 같다. 

### Policy Objective Functions

주어진 policy $\pi_{\theta}$(s,a) 에 대해서 제일 좋은 $\theta$(이하 세타)를 찾아야한다. 여기서 policy는 특정 파라미터 세타에 대해서 action a를 할 확률을 던져주는 함수이다. 이 policy $\pi_{\theta}$를 평가하기 위해서 목적함수를 만들어야한다. 목적함수 기준은 최종 목표인 reward를 가장 많이 받는 것으로 정할 수 있다. 

그렇다면 reward의 기대값인 value 값을 알면 value값이 높을 경우 좋은 policy가 된다고 할 수 있다. 

1. 에피소드로 나눠지는 환경일 경우 start value를 사용한다.
    - 처음 시작 (state) value를 기준으로 policy $\pi_{\theta}$를 따랐을 때 얼마의 reward를 받을 지(얼마의 value 기대값)에 대한 함수를 목적함수로 한다.
2. 연속 되는 환경에서는 average value를 사용한다.
    - 연속 되는 환경에서는 각 state에서 있을 확률 분포$d^{\pi_\theta}$(stationary distribution으로 각 state에 agent가 있을 확률이 어느정도 수렴하기 때문에 그 분포를 알 수 있다)와 그 state에서의 value를 곱한 값의 합을 목적함수로 한다.
3. 연속 된 환경에서는 average reward per time-step을 사용한다.
    - 앞선 방법과 비슷한데, 각 state에서의 value값을 stationary distribution에 곱했다면 여기서는 각 state에서의 policy와 Reward를 곱하여 합하고 이를 stationary distribution과 곱한다..이는 나중에 다시 공부해보자.

    좋은 점은 어떤 방법론이 있다면 위의 목적함수 3가지를 모두 최적화 한다.

### Policy Optimization

J(theta)에서 J값을 최대로 만들어주는 세타를 찾는게 우리의 목표다. 이때 policy는 세타로 파라미터화 되어 있기 때문에 세타는 policy를 정해준다고 볼 수 있다. policy를 어떻게 정해주냐에 따라서 reward가 바뀌고 그게 결국 J를 바꾸게 된다. 

### Computing Gradients By Finite Differences

최적화 문제를 풀기 위해서 Gradient 문제를 풀어야 한다. 여기서는 세타를 조금씩 바꿔가면서 전체 Gradient를 구하는 방식을 설명한다. 세타1~n까지 있을 때 세타값을 조금씩 바꿔서 그 difference로 기울기를 구해서 모으면 하나의 Gradient(벡터)가 된다. 다만, n차원일 경우 n번을 평가해야하므로 느리고 비싸다. 

### Score Function

Policy gradient를 해석적으로 계산한다. policy $\pi_\theta$가 0이 아닌 경우에는 미분 가능하다고 가정한다. 

그리고 $\pi_\theta$의 그라디언트는 안다고 가정한다.  이를 수식으로 표현하게 되면

$\bigtriangledown_\theta \pi_\theta(s,a) =\pi_\theta(s,a) \cfrac{\bigtriangledown_\theta \pi_\theta(s,a)}{\theta \pi_\theta(s,a)}=\pi_\theta(s,a)\bigtriangledown_\theta log\pi_\theta(s,a)$

수식을 생각해보면 log pi를 미분하면 1/pi가 된다. 이걸 이용해서 식을 정리하면 위 식과 같이 나온다. 이런 수식을 조작하는 이유는 추후에 편하게 하기 위함인데 이해를 위해 one step MDP를 고려해보자. one step MDP는 one step 후 reward를 받고 끝나는 것이다. one step MDP의 목적함수를 보자.

$J(\theta)=E_{\pi_\theta}[r]=\sum d(s)=\sum \pi_\theta(s,a)R_{s,a}$

각 initial state의 분포가 주어지면, state에서 어떤 action을 할 지에 대한 확률과 그때의 Reward를 곱한다. 목적함수를 미분해보면 아래와 같아진다.

![Lecture7%20136df1af18334ad6bbb34e91bdbbabaf/Untitled%201.png](Lecture7%20136df1af18334ad6bbb34e91bdbbabaf/Untitled%201.png)

위에서 사용했던 방법으로 그라디언트 파이를 바꾸면 그라디언으 log파이가 된다. 목적함수의 경우 기대값이다. 이 생각을 가지고 위 식을 보게 되면

J의 그라디언트가 policy 뒤에 있는 그라디언트 log pi의 기대값이 되게 된다.  

### Softmax Policy