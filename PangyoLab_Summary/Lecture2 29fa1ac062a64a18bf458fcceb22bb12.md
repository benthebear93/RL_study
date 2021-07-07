# Lecture2

### Markov Decision Processes

MDP는 RL에서 주어지는 환경을 이야기한다. 이때 환경은 관찰할 수 있으며, 현재 state가 process를 완벽히 특징화한다. 대부분의 RL 문제들이 MDP 형태로 가공이 가능하다. 최적제어의 경우 연속적인 MDP를 주로 다루며, Partially observable 문제도 MDP로 변환이 가능하다.  

현재 state인 St는 과거의 정보들 (S1~St)을 모두 포함하고 있다고 볼 수 있다. 따라서 현재 state 상태를 안다면 과거 정보는 필요 없어진다. 

현재 상태 St는 $P[S_{t+1}|St]=P[S_{t+1}|S_1,...,S+t]$을 만족해야만 Markov라고 부른다.

### State Transition Matrix

Markov state s와 successor state s'에 대해 state transition probability는 $P_{ss'}=P[S_{t+1}=s'|S_t=s]$로 정의 된다.

상태변환 매트릭스는 state s에서 successor state s'로 가는 모든 확률을 표현하는 매트릭스이다. 

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled.png)

### Markov Process

Markov process는 메모리가 없는 랜덤한 프로세스로 랜덤한 state들에 연속으로 표현된다. **각 state들은 다른 state로 넘어갈 때의 확률이 정해져 있고, 이를 <S,P>로 표현된다.**

S는 state, P는 state Transition Matrix이다.

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%201.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%201.png)

학생의 행동 패턴을 Markov chain 혹은 Markov process로 나타낸 사진이다. 

이때 순차적인 state들의 모음을 episode라고 한다.

예를 들어 학생의 경우 

c1 -c2-c3-pass-sleep

c1-FB-FB-C1-C2-Sleep이 될 수도 있는 것이다.

이런 state에서 state들을 왔다갔다 하는 확률을 변환 매트릭스로 정리하면 아래와 같다.

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%202.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%202.png)

### Markov Reward Process

기본적인 Markov Process가 각 state들과 확률이 정해져 있었다면, MRP는 각 state들에 대한 Reward가 정해져 있는 형태이다. MRP는 <S,P,R,$\gamma$>로 표현 된다. 

R 은 Reward 함수로 $R_s = E[R_{t+1}|S_t=s]$ 평균값으로 표현 된다. $\gamma$의 경우 discount factor로 0~1사이 값이다.

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%203.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%203.png)

### Return

reward와 return은 다른 개념으로, Return은

 $G_t = R_{t+1}+\gamma*R_{t+2}+...=\sum r^kR_{t+k+1}$

로 표현되며, reward를 discount한 값을 모두 더한 것이다. 

### Discount

Markov reward와 decision processes에서는 discount를 하는데 그 이유는 reward를 discount하는게 수학적으로 편리해서이다. 또한 MP가 무한정으로 돌아가는 것을 막기 위해서이다. 또한 미래에 대한 불확실성이 표현되지 않기 때문에 reward에 discount를 해준다. 경제적으로 고려했을 때 바로 앞에 들어오는(빨리 들어오는) reward가 더욱 효과적이고 나중에 들어오는 reward는 그렇지 못하기 때문이다. 

### Value Function

MRP의 value 함수는 v(s)로 표시되고 이는 state s에서 return의 기대값이다. $v(s) = E[G_t|S_t=s]$ 로 표시된다.

### Bellman Equation for MRPs

value 함수는 2개의 부분으로 나눠질 수 있다.

즉각적인 reward인 $R_{t+1}$과 successor state의 discounted 된 $\gamma*v(S_{t+1})$

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%204.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%204.png)

$v(s) = R_s+\gamma*\sum P_{ss'}v(s')$

조금 더 포멀하게 정리하면

$v = R+\gamma*P*v$

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%205.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%205.png)

### Bellman equation solving

$v =(I-\gamma*P)^-1*R$

다만 이런 형태의 Bellman equation 풀이는 작은 MRP에서는 가능하다. 큰 형태의 MRP를 푸는 데는 DP, Monte-Carlo, TD와 같은 다양한 방법들이 있다.

### Markov Decision Process

MDP는 Markov reward process와 decision이 합쳐진 형태이다. MRP와 다른 점을 조금 찾아보면, MRP에서는 state들에 이름이 있고 state에 도착하면서 reward를 받는 형태이다. MDP는 이와 조금 다르게 어떤 action을 하게 되면 그에 따른 reward를 받게 된다. 이때 환경에서 모든 state는 Markov하다. 

MDP는 <S,A,P,R,$\gamma$>로 표현 된다. A는 action의 집합이다.

MRP랑 다르게 식이 조금 바뀌게 된다.

$P_{ss'}^a=P[S_{t+1}=s'|S_t=s,A_t=a]$

$R_s = E[R_{t+1}|S_t=s,A_t=a]$

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%206.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%206.png)

### Policies

policy $\pi$는 주어진 state에서의 action을 이야기 한다. policy는 agent의 행동을 지정하는데, MDP에서의 policy는 현재 state랑 관련이 있지 history와는 관련이 없다. 즉 policy는 시간과 관련이 없이 독립적이다. 

MDP M = $<S,A,P,R,\gamma>$와 $\pi$에서 state의 sequence는 Markov process $<S,P^\pi>$이고 여기에 reward를 추가한 equence는 $<S,P^\pi, R^\pi, \gamma>$이다.

policy가 포함 된 s의 Probablity와 Reward를 정리하면 아래와 같다.

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%207.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%207.png)

즉, 특정 state에 대한 action의 policy와 각각 P와 R를 곱해 더하면 된다.

state-value function의 경우 state s에서 시작해서 policy를 따라서 나오는 return 값들의 기대값이다.

$v_\pi(s)=E_\pi[G_t|S_t=s]$

action-value function의 경우 state s에서 시작해여 action a를 실행하고 policy 를 따라서 나오는 return 값들의 기대값이다.

$q_\pi(s,a)=E_\pi[G_t|S_t=s,A_t=a]$

### Bellman Expectation Equation

state-value 함수는 immediate reward와 successor state가discount된 값의 형태로 표현이 가능하다.

$v_\pi(s)=E_\pi[R_{t+1}+\gamma*v_\pi(S_{t+1})|S_t=s]$

action-value함수 또한 비슷한 형태로 표현 가능하다.

$q_\pi(s,a)=E_\pi[R_{t+1}+\gamma*q_\pi(S_{t+1},A_{t+1})|S_t=s,A_t=a]$

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%208.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%208.png)

식을 매트릭스 형태로 포멀하게 정리해보면

$v_\pi=R^\pi+\gamma P^\pi v_\pi$

$v_\pi= (I-\gamma P^\pi v_\pi)^{-1}R^\pi$

### Optimal value function

optimal state value function은 v*(s)로 표현 되는데, 모든 policies에 대해 최대 value function을 말한다.

$v_*(s)=\max v_\pi(s)$

optimal action value function은 q*(s,a)로 모든 policies에 대해 최대 action value function을 말한다.

$q_*(s,a)=\max q_\pi(s,a)$

ㅐ

optimal value function에서는 MDP에서 찾을 수 있는 최고의 performance를 알려주고, 이 optimal value function을 알게 되면 MDP를 풀었다 라고 할 수 있다.

### Optimal policy

$\pi>=\pi' if v_\pi(s)>=v_\pi'(s), all-s$

정리하자면, 만약 state에서 프라임 policy보다 일반 policy의 value 값이 더 크거나 같다면, 프라임 policy보다 일반 policy도 크거나 같다는 뜻이다.

어떤 MDP이던 일반 policy보다 더 좋거나 같은 optimal policy인 $\pi_*$가 존재한다.

이때 optimal policies는 optimal value function과 같은 값을 갖게 된다.

optimal policies들은 optimal action-value function 또한 갖게 된다.

optimal policy는 optimal action value function인 $q_*(s,a)$를 최대로 만들어서 찾을 수 있다.

![Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%209.png](Lecture2%2029fa1ac062a64a18bf458fcceb22bb12/Untitled%209.png)

### Bellman optimality equation

벨만 최적식은 비선형이라서 앞서 본 벨만식들처럼 풀 수 없다. 따라서 반복해서 솔루션을 찾아야되는데 그 방법으로 value iteration, policy iteration, Q-learning, Sarsa 등이 있다.