import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque


def test(args, model, env):
    torch.manual_seed(args.seed)

    # env = create_atari_env(args.env_name)
    # env = create_car_racing_env()
    env.seed(args.seed)
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        #env.render()
        episode_length += 1
        # Sync with the shared model
        if done:
            # model.load_state_dict(shared_model.state_dict())
            hx = Variable(torch.zeros(1, 256))
        else:
            hx = Variable(hx.data)

        value, logit, hx = model((Variable(state.unsqueeze(0)), hx))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1)[1].data.numpy()

        state, reward, done, _ = env.step(action[0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        # actions.append(action[0])
        # if actions.count(actions[0]) == actions.maxlen:
        #     done = True

        if done:
            # print("Time {}, episode reward {}, episode length {}".format(
            #     time.strftime("%Hh %Mm %Ss",
            #                   time.gmtime(time.time() - start_time)),
            #     reward_sum, episode_length))
            # reward_sum = 0
            # episode_length = 0
            actions.clear()
            state = env.reset()
            return reward_sum, episode_length
            # time.sleep(60)

        state = torch.from_numpy(state)
