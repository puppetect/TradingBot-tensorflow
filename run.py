import gym
import envs


if __name__ == '__main__':
    env = gym.make('stock-v0')
    obs = env.reset()
    buffer = []
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        buffer.append([obs, reward, done, info])
    print(buffer)
