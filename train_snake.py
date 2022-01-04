from lib.dqn import train
from game.snake import SnakeGameAI
import gym
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class SnakeEnv(object):

    def __init__(self, speed=40):
        self.game = SnakeGameAI(speed=speed)
        self.action_space = gym.spaces.Discrete(n=3)

    def reset(self):
        self.game.reset()
        return self.game.get_state()

    def step(self, action):
        move = [0, 0, 0]
        move[action] = 1
        reward, is_done, score = self.game.play_step(move)
        new_state = self.game.get_state()
        return new_state, reward, is_done, score


if __name__ == "__main__":
    env_name = "Snake"
    env = SnakeEnv(speed=400)
    obs_size = 11
    n_actions = 3
    hidden_size = 256
    print('obs size', obs_size, 'n_actions', n_actions)
    net = Net(obs_size, hidden_size, n_actions)
    tgt_net = Net(obs_size, hidden_size, n_actions)

    train(env, net, tgt_net, env_name=env_name, EPSILON_DECAY_LAST_FRAME=20000)
