import time
import numpy as np

import collections
import torch
from train_snake import Net, SnakeEnv

FPS = 25

if __name__ == "__main__":

    env_name = "Snake"
    env = SnakeEnv()
    obs_size = 11
    n_actions = 3
    hidden_size = 256
    net = Net(obs_size, hidden_size, n_actions)
    net.load_state_dict(torch.load(f'{env_name}-best.dat', map_location=lambda storage, loc: storage))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
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

