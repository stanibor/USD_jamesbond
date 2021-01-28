import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os
import cv2
#from scipy.misc import imresize
import matplotlib.pyplot as plt

from ddqn_model import Atari_Wrapper, DQN, Agent

env_name = 'Jamesbond-v0'

total_episodes = 14
eval_epsilon = 0.01
num_stacked_frames = 4

#while training on colab
device = torch.device("cuda:0")
#while training on laptop/PC without NVIDIA GPU
#device = torch.device("cpu:0")
dtype = torch.float

f = open(env_name+"-Eval"+".csv","w")
f.write("steps,return\n")

for filename in os.listdir():

    if env_name not in filename or ".pt" not in filename:
        continue

    print("load file name", filename)
    agent = torch.load(filename, map_location='cpu').to(device)
    agent.set_epsilon(eval_epsilon)
    agent.eval()

    raw_env = gym.make(env_name)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)



    ob = env.reset()
    num_episode = 0
    returns = []
    while num_episode < total_episodes:

        action = agent.e_greedy(torch.tensor(ob, dtype=dtype).unsqueeze(0).to(device) / 255)
        action = action.detach().cpu().numpy()

        ob, _, done, info, _ = env.step(action, render=True)

        #time.sleep(0.016)

        if done:
            ob = env.reset()
            returns.append(info["return"])
            num_episode += 1

    env.close()

    steps = filename.strip().split(".")[0].split("-")[-1]
    f.write(f'{steps},{np.mean(returns)}\n')

f.close()