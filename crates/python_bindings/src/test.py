from rust_drl import test
import gymnasium as gym

env = gym.make('CartPole-v1')
a = env.reset(seed=0)
print(a)
a = env.reset(seed=0)
print(a)

r = test(env)
print(r)
r = test(env)
print(r)
r = test(env)
print(r)

