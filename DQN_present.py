import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os

from ddqn_model import Atari_Wrapper, DQN, Agent
import cv2

dtype = torch.float
num_stacked_frames = 4
eval_epsilon = 0.01
filename = 'best_model.pt'
FPS = 120

device = torch.device('cpu:0')

cv2.namedWindow('agent vision', cv2.WINDOW_NORMAL)
cv2.resizeWindow('agent vision', 600, 600)

if __name__ == '__main__':
    env_name = 'Jamesbond-v0'
    env = gym.make(env_name)

    agent = torch.load(filename, map_location='cpu').to(device)
    agent.set_epsilon(eval_epsilon)
    agent.eval()

    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)

    ob = env.reset()
    num_episode = 0
    returns = []
    while True:
        try:
            action = agent.e_greedy(torch.tensor(ob, dtype=dtype).unsqueeze(0).to(device) / 255)
            action = action.detach().cpu().numpy()

            ob, _, done, info, _ = env.step(action, render=True)

            cv2.imshow('agent vision', ob[-1])
            cv2.waitKey(int(1000/FPS))

            if done:
                ob = env.reset()
                print(f'Score: {info["return"]}')
        except KeyboardInterrupt:
            print('Bye')
            env.close()
            break

