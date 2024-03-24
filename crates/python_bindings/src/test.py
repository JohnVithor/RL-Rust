from rust_drl import test
import gymnasium as gym

env = gym.make('CartPole-v1')
(obs, info) = env.reset(seed=0)
(obs, reward, terminated, truncated, info) = env.step(0)
print((obs, reward, terminated))

#####################################
r = test(env)
print(r)

