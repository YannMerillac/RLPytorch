from lib.dqn import train
from lib import wrappers
import gym
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    env_name = "Breakout-ram-v0"
    env = gym.make(env_name)
    env = wrappers.MaxAndSkipEnv(env)
    env = wrappers.FireResetEnv(env)
    env = wrappers.BytesToBits(env)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print('obs size', obs_size, 'n_actions', n_actions)
    net = Net(obs_size, n_actions)
    tgt_net = Net(obs_size, n_actions)

    train(env, net, tgt_net, env_name=env_name)
