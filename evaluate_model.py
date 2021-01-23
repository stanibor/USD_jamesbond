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

env_name = 'Jamesbond-v0'

total_episodes = 14
eval_epsilon = 0.01
num_stacked_frames = 4

#while training on colab
device = torch.device("cuda:0")
#while training on laptop/PC without NVIDIA GPU
#device = torch.device("cpu:0")
dtype = torch.float

class Atari_Wrapper(gym.Wrapper):

    # env wrapper to resize images, grey scale and frame stacking and other misc.
    def __init__(self, env, env_name, k, dsize=(84,84), use_add_done=False):
        super(Atari_Wrapper, self).__init__(env)
        self.dsize = dsize
        self.k = k
        self.use_add_done = use_add_done

        # set image cutout depending on game
        if "Pong" in env_name:
            self.frame_cutout_h = (33,-15)
            self.frame_cutout_w = (0,-1)
        elif "Breakout" in env_name:
            self.frame_cutout_h = (31,-16)
            self.frame_cutout_w = (7,-7)
        elif "SpaceInvaders" in env_name:
            self.frame_cutout_h = (25,-7)
            self.frame_cutout_w = (7,-7)
        elif "Jamesbond" in env_name:
            self.frame_cutout_h  = (38,-20)
            self.frame_cutout_w = (8,-1)
        else:
            # no cutout
            self.frame_cutout_h = (0,-1)
            self.frame_cutout_w = (0,-1)

    def reset(self):

        self.Return = 0
        self.last_life_count = 0

        ob = self.env.reset()
        ob = self.preprocess_observation(ob)

        # stack k times the reset ob
        self.frame_stack = np.stack([ob for i in range(self.k)])

        return self.frame_stack


    def step(self, action, render=False):
        # do k frameskips, same action for every intermediate frame
        # stacking k frames

        reward = 0
        done = False
        additional_done = False

        # k frame skips or end of episode
        frames = []
        for i in range(self.k):

            ob, r, d, info = self.env.step(action)
            if render:
                self.render()
                time.sleep(0.004)

            # insert a (additional) done, when agent loses a life (Games with lives)
            if self.use_add_done:
                if info['ale.lives'] < self.last_life_count:
                    additional_done = True
                self.last_life_count = info['ale.lives']

            ob = self.preprocess_observation(ob)
            frames.append(ob)

            # add reward
            reward += r

            if d: # env done
                done = True
                break

        # build the observation
        self.step_frame_stack(frames)

        # add info, get return of the completed episode
        self.Return += reward
        if done:
            info["return"] = self.Return

        # clip reward
        if reward > 0:
            reward = 1
        elif reward == 0:
            reward = 0
        else:
            reward = -1

        return self.frame_stack, reward, done, info, additional_done

    def step_frame_stack(self, frames):

        num_frames = len(frames)

        if num_frames == self.k:
            self.frame_stack = np.stack(frames)
        elif num_frames > self.k:
            self.frame_stack = np.array(frames[-self.k::])
        else: # mostly used when episode ends

            # shift the existing frames in the framestack to the front=0 (0->k, index is time)
            self.frame_stack[0: self.k - num_frames] = self.frame_stack[num_frames::]
            # insert the new frames into the stack
            self.frame_stack[self.k - num_frames::] = np.array(frames)

    def preprocess_observation(self, ob):
    # resize and grey and cutout image

        ob = cv2.cvtColor(ob[self.frame_cutout_h[0]:self.frame_cutout_h[1],
                           self.frame_cutout_w[0]:self.frame_cutout_w[1]], cv2.COLOR_BGR2GRAY)
        ob = cv2.resize(ob, dsize=self.dsize)

        return ob

class DQN(nn.Module):
    # nature paper architecture

    def __init__(self, in_channels, num_actions):
        super().__init__()

        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        ]

        self.network = nn.Sequential(*network)

    def forward(self, x):
        actions = self.network(x)
        return actions

class Agent(nn.Module):

    def __init__(self, in_channels, num_actions, epsilon):
        super().__init__()

        self.in_channels = in_channels
        self.num_actions = num_actions
        self.network = DQN(in_channels, num_actions)

        self.eps = epsilon

    def forward(self, x):
        actions = self.network(x)
        return actions

    def e_greedy(self, x):

        actions = self.forward(x)

        greedy = torch.rand(1)
        if self.eps < greedy:
            return torch.argmax(actions)
        else:
            return (torch.rand(1) * self.num_actions).type('torch.LongTensor')[0]

    def greedy(self, x):
        actions = self.forward(x)
        return torch.argmax(actions)

    def set_epsilon(self, epsilon):
        self.eps = epsilon

f = open(env_name+"-Eval"+".csv","w")
f.write("steps,return\n")

for filename in os.listdir():

    if env_name not in filename or ".pt" not in filename:
        continue

    print("load file name", filename)
    agent = torch.load(filename).to(device)
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