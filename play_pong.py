#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np

import collections
import torch
from lib import wrappers
from train_pong import Net

DEFAULT_ENV_NAME = "Pong-ram-v0"
FPS = 25


if __name__ == "__main__":

    env_name = "Breakout-ram-v0"
    env = gym.make(env_name)
    env = wrappers.FireResetEnv(env)
    #env = wrappers.MaxAndSkipEnv(env)
    env = wrappers.BytesToBits(env)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, n_actions)
    net.load_state_dict(torch.load(f'{env_name}-best.dat', map_location=lambda storage, loc: storage))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v.float()).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)

