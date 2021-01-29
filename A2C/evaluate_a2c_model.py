import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
import random
import os
import cv2
#from scipy.misc import imresize
import matplotlib.pyplot as plt

from model import ActorCritic
from envs import create_jamesbond_env

env_name = 'Jamesbond-v0'

total_episodes = 14
eval_epsilon = 0.01
num_stacked_frames = 4

#while training on colab
# device = torch.device("cuda:0")
#while training on laptop/PC without NVIDIA GPU
dtype = torch.float
models_dir = 'models'
render = True

f = open("A2C-"+env_name+"-Eval"+".csv","w")
f.write("steps,return\n")

for filename in os.listdir(models_dir):

    if ".pkl" not in filename:
        continue


    print("load file name", filename)

    fullfilename = os.path.join(models_dir, filename)
    env = create_jamesbond_env(env_name)
    ob = env.reset()

    agent = ActorCritic(env.observation_space.shape[0], env.action_space)
    agent.load_state_dict(torch.load(fullfilename))
    hx = Variable(torch.zeros(1, 512))

    agent.eval()

    num_episode = 0
    reward_sum = 0
    returns = []
    while num_episode < total_episodes:

        ob = torch.from_numpy(ob).float()
        value, logit, hx = agent((Variable(ob.unsqueeze(0)), hx))
        probability = F.softmax(logit, dim=-1)
        action = probability.max(1)[1].data.numpy()

        ob, reward, done, info = env.step(action)
        if render:
            env.render()

        #time.sleep(0.016)
        reward_sum += reward

        if done:
            ob = env.reset()
            returns.append(reward_sum)
            reward_sum = 0
            num_episode += 1
            hx = Variable(torch.zeros(1, 512))

    env.close()

    steps = filename.strip().split(".")[0].split("_")[-1]
    f.write(f'{steps},{np.mean(returns)}\n')

f.close()