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

from model import ActorCritic
from envs import create_jamesbond_env
import cv2

dtype = torch.float
filename = 'best_model.pkl'
FPS = 120

device = torch.device('cpu:0')

cv2.namedWindow('agent vision', cv2.WINDOW_NORMAL)
cv2.resizeWindow('agent vision', 600, 600)

if __name__ == '__main__':
    env_name = 'Jamesbond-v0'
    env = create_jamesbond_env(env_name)



    agent = ActorCritic(env.observation_space.shape[0], env.action_space)
    agent.load_state_dict(torch.load(filename))
    hx = Variable(torch.zeros(1, 512))


    ob = env.reset()
    num_episode = 0
    reward_sum = 0
    returns = []
    while True:
        try:
            ob = torch.from_numpy(ob).float()
            value, logit, hx = agent((Variable(ob.unsqueeze(0)), hx))
            probability = F.softmax(logit, dim=-1)
            action = probability.max(1)[1].data.numpy()

            ob, reward, done, info = env.step(action)
            env.render()

            reward_sum += reward

            cv2.imshow('agent vision', ob[-1])
            cv2.waitKey(int(1000/FPS))

            if done:
                ob = env.reset()
                print(f'Score: {reward_sum}')
                reward_sum = 0
        except KeyboardInterrupt:
            print('Bye')
            env.close()
            break

