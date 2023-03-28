import gym

from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    if done:
      break

env.close()
print(f'Episode reward: {info["episode"]["r"]}')