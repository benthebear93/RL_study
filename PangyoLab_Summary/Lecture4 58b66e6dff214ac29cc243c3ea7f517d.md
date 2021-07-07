# Lecture4

### Model Free Reinforcement Learning

### Monte-Carlo Reinforcement Learning

MC방법은 모델 Free 이기 때문에 MDP(state 이동, reward) 에 대해 정보가 없고, 에피소드 자체의 경험에서 직접 배우는 방식이다. bootstrapping 없이 완벽한 에피소드를 통해 학습된다. 즉 에피소드가 끝날 때마다 return 값을 적어둔 후 평균을 value값으로 정한다. 단점은 당연하게도 에피소드가 끝나야 적용된다는 점이다.

### Monte-Carlo Policy Evaluation

$\pi$ policy를 따르는 에피소드를 통한 $v_{\pi}$를 구하는 것이 목적이다. 

이야기 하기 전에 앞서 discount 된 reward와 value function을 다시 한번 보고 가자.

$G_t = R_{t+1}+\gamma*R_{t+2}+...=\sum r^kR_{t+k+1}$ **(Reward)**

$v(s) = E[G_t|S_t=s]$ **(value function)**

MC Policy evaluation의 경우 return의 기대값을 사용하는 게 아닌, 경험적(empirical) return 평균을 사용한다. 

### First Visit Monte-Carlo Policy Evaluation

state s를 평가하는 방법으로 한 에피소드안에 state가 여러번 있고 에피소드가 진행되면서 state들을 여러번 방문하게 된다. 이때 처음 방문할 때만 state에 대한 count를 올려주고 나머지는 올리지 않는 방법이다. 그래서 처음 방문할 때만 return을 더해주고 이들의 평균을 낸다. law of large numbers에 의해서 count값인 N이 무한대로 가게 되면 기대값이니 value함수 값이 수렴하게 된다.

### Every-Visit Monte-Carlo Policy Evaluation

First visit MC Policy evaluation과 알고리즘 자체는 동일하지만 state를 처음 방문할 때만 counter를 더해주는 게 아니라서 방문 할때 마다 올려준다. 

다만 이 2가지 방법들은 모든 state를 방문한다는 조건이 있다. 결국 N이 무한대로 가야되기 때문이다. state를 방문하지 않으면 N=0으로 value값이 없어지기 때문이다.

### Incremental Mean

MC는 여러번 한 후 평균을 내야한다. 이때 평균을 조금 다른 표현할 수 있는데 이는 k번째와 k-1번째와의 차이를 이용해서 표현하는 방식이다. 

$\mu_k=(1/k)\sum^k x_j =(1/k)(x_k+\sum^{(k-1)} x_j)$

$=(1/k)(x_k+(k-1)\mu_{k-1})$

$=\mu_{k-1}+(1/k)(x_k-\mu_{k-1})$

### Incremental Monte-Carlo Updates

Incremental Mean 표현 방식과 동일하게  value 함수 V(s)도 Incrementally update할 수 있다. state $S_t$와 return $G_t$를 이용하면

$N(S_t) \leftarrow N(S_t)+1$

$V(S_t)\leftarrow V(S_t)+1/N(S_t)(G_t-V(S_t))$

이처럼 표현되는데 생각해보면 현재 나온 return 값 Gt와 Value값의 차이만큼 (에러만큼) 업데이트 해준다고 볼 수 있다.

여기서 N의 경우 시간이 지날 수록 커지게 되고 에러값에 곱해지는 숫자 자체는 작아지게 되는데 이를 $\alpha$로 고정할 수도 있다. 고정하게 되면 예전의 값들을(히스토리를?) 무시하게 된다. 조금 더 생각해보면, G에서 V를 빼서 V에 더해주면서 업데이트 하니깐 V가 G에 가까워 지도록 업데이트 한다고 생각할 수도 있다.

### Temporal-Difference Learning

TD는 에피소드로부터 바로 학습되는데, MC와 동일하게 MDP의 Transition이나 Reward가 없어도 가능하다. TD는 끝나지 않은 에피소드에서도 bootstrapping을 통해서 학습할 수 있다. 

 TD는 guess로 guess를 업데이트 한다고 하는데, 직관적으로 이해하기 위해서 MC와 비교해보자.

MC의 경우 업데이트 식이 아래와 같다.

$V(S_t)\leftarrow V(S_t)+1/N(S_t)(G_t-V(S_t))$

TD의 경우 업데이트 식은 아래와 같은데

$V(S_t)\leftarrow V(S_t)+\alpha(R_{t+1}+\gamma V(S_{t+1})-V(S_t))$

현재 위치인 St에 대한 정보가 아닌 다음 위치인 St+1이 들어가 있는 걸 볼 수 있다.

$R_{t+1}+\gamma V(S_{t+1})$는 TD Target이라고 하며 이를 사용하는 이유는 당연하게도(?) 한 스텝을 더 나아가 예측하는 게 더 정확하기 때문이다. 한 스텝을 더 나아가 예측하는 방향으로 V를 업데이트 하는 것이다.즉, TD Target이라는 guess와 현재 V값의 차이로 V값을 업데이트 해준다. 

### Advantages and Disadvantages of MC vs TD

TD는 끝나기 전에 학습을 할 수 있다. Online으로 매 스텝마다 학습을 할 수 있고 끝나지 않은(혹은 완벽하지 않은) sequence를 통해서도 학습이 가능하지만 MC는 무조건 한 에피소드가 끝나서 return 값이 나와고 완벽한 sequence를 통해서만 학습이 가능하다. 즉 TD는 진행되는 환경에서 학습하고 MC는 에피소드 처럼 분리되서 끝나는 환경에서만 적용이 가능하다. 

### Bias/Variance Trade off

일반적인 Gt의 경우 에피소드가 이루어지는 과정에서 많은 변수들에 따라 다양한 값들이 나오게 된다. 그렇게 Gt의 분포는 variance가 큰 분포가 되지만 결과적으로는 $v_{\pi}(S_t)$에 수렴하게 된다. 이와 반대로 True TD target은 바로 한 스텝 앞을 보고 값을 수정하게 된다. 바로 한 스텝이니 여러 변수들이 아닌 1개씩만 영향을 미치게 되고, 이는 low variance를 만들다. 하지만 $v_{\pi}(S_t)$는 biased estimate이 된다. 

MC와 TD를 이와 같은 개념들을 바탕으로 비교해보자.

- MC는 variance가 크고, bias가 없다
    - 좋은 수렴성
    - 초기값에 민감하지 않음
    - 이해가 쉽고 간단히 사용 가능
- TD는 variance가 작고, bias가 조금 있다.
    - 주로 MC보다는 효율적이다
    - TD(0)은 $v_{\pi}(s)$로 수렴하지만 approximation을 하면  항상 그런건 아니다.
    - 초기값에 민감하다.

### Batch MC, TD

무한대로 에피소드를 진행하면 MC랑 TD는 수렴한다는 걸 알았다. 그렇다면 무한번이 아닌 K번 만큼만 해도 수렴할까?

### Certainty Equivalence

MC의 해는 단순히 mean-squared error를 줄이는 방향으로 구해진다. 에피소드 끝에서 관찰된 return에 맞는 방향으로 간다. 

$\sum^k \sum^{T_k}(G^k_t-V(s_t^k))^2$

TD(0)은 likelihood Markov model의 max값으로 수렴한다.

MDP에서 주어지는 S,A,P,R,$\gamma$에 가장 잘 맞는 해로 수렴하게 된다. 

$\hat{P^a_{s,s'}}=1/N(s,a)\sum^K\sum^T_k 1(s_t^k,a_t^k,s_{t+1}^k=s,a,s')$

$\hat{R^a_{s,s'}}=1/N(s,a)\sum^K\sum^T_k 1(s_t^k,a_t^k=s,a)r_t^k$

이를 다르게 설명하면, TD는 Markov property를 사용해서 추측을 하기에 Markov 환경에서 더 효율적이고, MC는 단순히 MSE를 줄이는 방향으로 추측하기에 Markov 환경이 아닌 곳에서 더 효율적이다. 

### Bootstrapping and Sampling

Bootstrapping은 추측치(estimate)가 포함된다.(depth 위주)

- MC는 bootstrap X
- DP, TD는 bootstrap O

Sampling은 sample들을 expectation(width 위주)

- MC와 TD는 sample을 한다
- DP는 sample을 하지 않는다.

### n-Step Prediction

TD를 설명할 때 한 스텝 앞선 state를 참고하여 값을 업데이트 해준다고 했다. 그렇다면 2,3 스텝 앞을 보면 안될까? 이런 이유 때문에 n step에 따라 나눌 수 있으며 최대 step의 경우 MC와 같아진다고 볼 수 있다.

n step에 대한 return값을 식으로 표현하면 아래와 같다.

$G^{n}_t=R_{t+1}+\gamma R_{t+2}+...+\gamma^{n-1}R_{t+n}+\gamma^{n}V(S_{t+n})$

ㅇ