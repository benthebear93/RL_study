import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import gym

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["framework"] = "torch"
trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")

for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)


# import model saved
trainer.import_model("my_weights.h5")
env = gym.make("CartPole-v0")
# run until episode ends

episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = trainer.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward

print("reward :", episode_reward)

