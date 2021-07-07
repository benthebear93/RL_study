# Lecture5

## Model Free Control

RL 에서는 Prediction과 Control 문제로 나뉘게 되는데, 여기서는 Control 문제에 대해서 다루게 된다. Prediction이 MDP가 주어지지 않았을 때 Value function을 estimate하는 거였다면, Control에서는 Value function을 optimize하는데 중점을 둔다. 

현재까지는 Table lookup방법을 사용하여 설명했다. state에 맞춰서 Table을 만들어서 도출할 수 있는 값을 채워넣으면서  업데이트 하는 방식으로 학습했었다. 단, state가 많아지면 Table lookup으로는 풀 수 없다. 따라서 function approximater가 value나 policy 들을 근사하는 방법이 존재한다.

### On policy Monte Carlo Control

on policy learning는 최적화 하고자 하는 policy와 환경에서 경험을 쌓는 policy가 같을 때를 말한다. "Learn on the job"

(복습) Generalised Policy Iteration

Policy Iteration은 2가지 단계로, Policy 평가 →Greedy 향상을 반복하는 걸로 이루어진다. 이는 결국 Control 방법론으로 Policy를 찾는 방법론, 최적의 value function을 찾는 방법론이다. 

그렇다면 이 방법을 On policy에 쓰면 되는거 아닐까? 라고 생각할 수 있다. 하지만 이 방법이 안되는데, MC를 통해 V를 학습하는데 이때 V를 학습하는 것은 Greedy를 통해 현재 state가 있고 다음 state를 알 때 제일 V값이 큰 곳으로 가는 Policy를 만드는 것이다. 여기서 다음 state를 안다는 것은 결국 MDP를 안다는 것이다. MDP를 모른다면 다음 state를 알 수 없다. 즉, policy evaluation은 가능하지만 policy imporvement인 Greedy policy improvement는 불가능하다. 

### Model-Free Policy Interation Using Action-Value Function

Greepy policy improvement는 V(s)를 사용하고 이는 MDP 모델이 필요하다. 반대로 state value function이 아닌 action value function인 $Q(s,a)$를 이용해서 improvement를 한다면 Model-free가 된다. 현재 state에서 어떤 action을 취할지는 MC를 통해 구할 수 있으니 이걸 사용하면 improvement를 할 수 있다. 여기서 improvement에서 Greedy 하게만 움직이면 state를 충분히 탐지하지 않아서 stuck될 수 있다. 

### epsilon-Greedy Exploration

입실론 그리디 방법은 충분한 탐지를 보장해주는 방식이다.

입실론의 작은 확률로 랜덤하게 다른 action을 선택하는 것이다. 이와 반대로 1-입실론의 확률로는 greedy한 action을 하게 된다. 이 방식으로 policy가 발전하면서 충분한 탐지를 보장해주게 된다. 

그렇다면 정말 그리디 방법 대신 입실론 그리디를 사용했을때 policy가 발전할까에 대한 증명은 아래 식으로 가능하다. 

$q_\pi(s,\pi'(s))=\sum \pi'(a|s)q_\pi(s,a)$

$=\epsilon/m \sum q_\pi(s,a)+(1-\epsilon)\max q_\pi(s,a)$

$>=\epsilon/m \sum q_\pi(s,a)+(1-\epsilon)((\pi(s|s)-\epsilon/m)/(1-\epsilon) q_\pi(s,a)=\sum \pi(a|s)q_\pi(s,a)=v_\pi(s)$

### Monte-Carlo Control

MC를 이용한 policy evaluation과 입실론 그리디를 이용한 improvement를 조금 더 효율적으로 하는 방법은 MC의 특징을 이용하는 것이다. MC는 한 에피소드가 끝나야 Value funtion을 업데이트 할 수 있는데, 수렴할 때까지 에피소드를 계속 돌리는 것이 아닌, 한 에피소드가 끝나면 바로 improvement로 넘어가는 방법이다.

### GLIE(Greedy in the limit with infinite Exploration)

1. 모든 state-action은 무한대로 explore되어야 한다.

    $\lim_{k-inf}N_k(s,a)=inf$

2. policy는 그리디 policy에서 수렴해야 한다.

    $\lim_{k-inf}\pi_k(a|s)=1(a=\argmax Q_k(s,a'))$

입실론 그리디를 사용하게 되면, 강제로 입실론 만큼 랜덤한 액션을 하기 때문에 이를 해결해야한다. 결국 입실론 그리디의 policy는 그리디 policy로 수렴해야한다.

따라서 입실론을 1/k 로 설정하여 시간이 지날 수록 0으로 수렴하도록 한다. 

### TD Control- Updating Action-Value Function with Sarsa

TD는 기본적으로 1스텝 단위로 업데이트가 가능하다. 따라서 state에서 action을 하고 Reward를 받고 업데이트 할 수 있다.

![Lecture5%200cab34eacd21470993dcbe1a54c204d1/Untitled.png](Lecture5%200cab34eacd21470993dcbe1a54c204d1/Untitled.png)

$Q(S,A)\leftarrow Q(S,A)+\alpha(R+\gamma Q(S',A')-Q(S,A))$

식을 음미해보면, $\alpha$가 얼마나 step을 넘어갈지를 정하고 그와 곱해진 것이 TD error 값으로 TD target과 현재 Q값의 차이가 된다. TD target은 한 스탭을 더 가서 예측 하는 예측치로 정해진다. 

### On-Policy Control With Sarsa

MC와는 다르게 한 에피소드마다가 아닌 한 step마다 evaluation과 improvement가 진행된다. 이때 evaluation은 Sarsa방식으로, improvement는 입실론 그리디 방식으로 진행된다. 

### Sarsa Algorithm for On policy Control

먼저 Q(s,a)를 임의값으로 업데이트 한다. 그 후 state에서 입실론 그리디에 맞게 그리디한 action을 취하거나, 랜덤하게 1-입실론 확률 값을 취한다. 그러면 Reward를 받고, 다음 state에서 입실론 그리디에 맞게 다시 action을 뽑는다. 이를 이용해서 Q(s,a)를 업데이트 한다.

Sarsa가 만족하기 위해서는 먼저 입실론 그리디를 사용하기 때문에 GLIE를 만족해야한다. 그리고 Robbins-Monro도 만족해야한다. R-M는 $\alpha$(step size)가 충분히 커서 Q를 큰 값으로도 수렴시킬 수 있어야한다는 조건이다. 또한 $\alpha^2$값이 무한대보다 작아야한다. 

                          $\sum^\infty_{t=1}\alpha_t=\infty$

                               $\sum^{\infty}_{t=1}\alpha^2_t<\infty$

### Backward View Sarsa($\lambda$)

TD($\lambda$)와 같이 eligibility traces를 사용한다. 책임을 뭍는 것으로 "최근에 방문 한 것" , "여러번 방문한 것"에 책임을 크게 주는 것을 고려하는 것이다. E는 각 state마다 가지고 있는 것으로 방문하면 1을 더해주고 시간이 지나감에 따라 $\gamma$를 곱해서 값을 감쇠해줬다. TD error $\delta_t$와 E값을 정리하면

$\delta_t=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)$

$Q(s,a)\leftarrow Q(s,a)+\alpha \delta_tE_t(s,a)$

Sarsa 람다는 기본적으로 state마다 eligibility trace를 남기고 그 값을 기록하고 그 크기에 따라 Q를 업데이트 하게 되므로 알고리즘에 있어서 한스탭마다 모든 Q를 업데이트하게 된다. 생각해보면 당연한게 방문했던 모든 state들은 eligibility trace값이 있기 때문이다. 

### Off-Policy Learning

Off policy는 앞전에 설명한 것 처럼 on policy가 최적화 하고자 하는 policy와 환경에서 쌓는 경험 policy가 같은 경우인 것과 반대로  behaviour policy $\mu$와 target policy $\pi$ 2개가 있고 서로 다를 경우를 말한다. 

우리는 policy를 evaluate하고 improvement를 하는 게 목적이다. 따라서 target policy를 evaluate 하려고(value function을 구하려고 v 혹은 q) 하는데 일반적인 $\pi$를 따라서 움직이는 것이 아닌 $\mu$를 따라서 움직이는 경우라고 해보자. 이런 경우에도 학습을 할 수 있는데, 다른 agent의 행동을 관찰하는 것만으로도 학습할 수 있는 것이다.  다른 agent의 policy를 따르는 것이 아닌 agent의 행동에 따른 reward를 참고하여 policy를 따를지 말지를 정할 수 있다는 것이다. 즉 경험에서 좋은 것만 가져가서 쓰겠다는거다. 

또한 on policy의 경우 경험이 끝나면 policy가 업데이트 되기 때문에 새로운 경험을 쌓아야 했지만, off-policy의 경우 agent가 동일하지 않을 수 있기 때문에 새롭지 않은 경험들을 가져다가 재사용할 수 있는 장점이 있다.  추가적으로 무언가 랜덤하고 과감한 행동들을 취하면서도 최적의 policy를 가져다가 학습할 수 있다는 장점이 있다. 

앞서 이야기한 입실론 그리디 방식의 문제점을 확실히 해결해주는 것으로, 랜덤한 액션을 취하면서도 최적의 경험을 통해 policy를 발전시킬 수 있다. 또한 다른 policy들을 한꺼번에 학습하는 것도 가능하다. 

### Importance Sampling

$E_{x-p}[f(x)]=\sum P(x)f(x)$

$=\sum Q(x)*(P(x)/Q(x))*f(x)$

=$E_{x-Q}[(P(x)/Q(x))*f(x)]$

Importance sampling은 식에서 P(x)를 이용해서 기대값을 구하려는데 그때 구하려는 기대값이 Q(x)일 때에 대한 솔루션을 이야기한다. p로 q를 구하고 싶다는 말인데, 이건 식을 조금씩 바꿔서 Q확률을 앞으로 빼면 구할 수 있게 된다. 즉, 뒤에 합이 무엇이 됐던 Q확률이 궁극적으로 곱해지면서 Q 확률에 대한 기대값이 나오게 된다는 말이다. 

2개의 다른 확률 분포가 있는데, A분포로 B분포의 기대값을 구하고 싶다면 두 경우의 비율을 곱해주면 된다. 

### Importance sampling for off-policy MC

offpolicy에서는 결국 $\mu$를 통해 얻어진 return을 이용해서 $\pi$를 평가하고 싶은 것이다. return $G_t$를 단순히 사용하는 것이 아니라 Importance sampling을 작용해서 Gt를 얻을 때까지 취했던 action들의 $\mu$와$\pi$의 확률들의 비율을 곱해준 값을 사용한다. 이렇게되면 $\mu$를 통해서도 $\pi$로 return 값을 얻은 것으로 수정된다는 것이다. (근데,,이는 현실에서 적용가능한 방법이 아니라고 한다. 그럼 왜 설명..?)

### Importance sampling for off-policy TD

MC가 불가능한 이유는 분수가 계속 곱해지면은 분모가 매우 커지거나 0이 될 수 있기 때문이다. TD의 경우 한스탭마다 사용하기 때문에 TD target에 sampling correction을 위한 분수를 한번만 곱해주면 된다.

$v(S_t)\leftarrow v(S_t)+\alpha(TD  target)-v(S_t))$

### Q-Learning

Q 러닝은 이름처럼 action-value 값을 off policy learning으로 학습하는 방법이다. 앞서 말한 importance sampling을 제외하고 사용해야 하는데, 먼저 behaviour policy $A_{t+1}-\mu(.|S_t)$를 사용해서 action을 하나 선택한다. 그 다음 도착한 $S_{t+1}$에 도착하여 예측할때 다시 behaviour policy를 사용하는게 아닌 traget policy $A'-\pi(.|S_t)$를 사용하면 된다. TD 랑 같은 원리지만 다음 state에서 behaviour policy를 사용해서 예측하는게 아니라 target policy를 사용하여 예측한다는 것만 다르다. 

$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha(R_{t+1}+\gamma Q(S_{t+1},A')-Q(S_t,A_t)$

### Off-Policy Control with Q-learning

behaviour policy도 imporve 시키고 싶은데, 또한 랜덤한(탐험적인) 행동들고 하고 싶다면 어떻게 해야할까? 앞에서 이야기 했던 입실론 그리디 방법으로 improvement하면 된다. 그러면 policy의 improvement를 보장하면서 랜덤한 행동도 하게 된다. 

정리하자면 Q learning은

- Target policy - greedy
- Behaviour policy - $\epsilon$  greedy

    $R_{t+1}+\gamma Q(S_{t+1},A')$

 =$R_{t+1}+\gamma Q(S_{t+1},\argmax Q(S_{t+1},a'))$

 =$R_{t+1}+\max\gamma Q(S_{t+1},a')$

![Lecture5%200cab34eacd21470993dcbe1a54c204d1/Untitled%201.png](Lecture5%200cab34eacd21470993dcbe1a54c204d1/Untitled%201.png)