**논문 제목 : Deterministic Policy Gradient Algorithms**

**논문링크 :** **http://proceedings.mlr.press/v32/silver14.pdf**

**https://github.com/benthebear93/RL_practice** **(추후 공개예정)**



**[Abstract]**

이번 논문은 continuous action에서의 강화학습을 위한 deterministic policy gradient  algorith에 대한 내용이다. DPG는 매력적인 형태를 가지고 있는데, action-value function의  gradient 기대값 형태라는 점이다. 이 간단한 형태는 stochastric policy gradient보다 더욱 효율적으로  추정 될 수 있다는 것을 말한다. 또한 적절한 탐색을 보장하기 위햐서  off-policy actor-critic 알고리즘을  소개한다. 마지막으로 논문에서는 DPG가 고차원 action space에서 SPG보다 더 좋은 선능을 낸다는 것을 보인다. 



**[Introduciton]**

PG 알고리즘의 기본 아이디어는 파라미터화 된 확률 분포로 정책을 표현하는 것이다. 이때 정책은 정책 파라미터에 따라 state에서  action을 stochastic하게 선택한다. 즉 PG 알고리즘은 일반적으로 stochastic한 정책에서 샘플링을 한 후 정책 파라미터를 높은 리워드를 만들 수 있게 수정하는 것이다. 



이전에는 model이 있거나 하지 않으면 dpg를 얻을 수 없다고 봤다. 하지만 dgp가 있고 이는 model free 형태로 간단하게  존재하면서, 이는 action-value function의 gradient의 형태를 따른다는 것을 알았다. 추가로 논문에서는  dpg가 spg의 특별 케이스인 policy의 gradient가 0에 수렴한다는 것을 보인다.

 

프렉티컬한 관점에서 봤을 때  spg와 dpg에는 큰 차이가 있다. stochastic의 경우 policy gradient가 state와  action 모두가 연관되어 있는데, deterministic의 경우 state space만 연관이 있다. 결과적으로 spg를  계산하는데 있어서 샘플이 더 필요하고, 만약 action-space가 차원이 클 경우 이는 심해진다. 