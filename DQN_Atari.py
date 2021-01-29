import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from ddqn_model import Atari_Wrapper, Agent, DQN
import random
import os
import cv2
#from scipy.misc import imresize
import matplotlib.pyplot as plt

# while training on colab
device = torch.device("cuda:0")
# while training on laptop/PC without NVIDIA GPU
# device = torch.device("cpu:0")
dtype = torch.float


class Logger:

    def __init__(self, filename):
        self.filename = filename

        f = open(f"{self.filename}.csv", "w")
        f.close()

    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()


class Experience_Replay():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transitions):

        for i in range(len(transitions)):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = transitions[i]
            self.position = (self.position + 1) % self.capacity

    def get(self, batch_size):
        # return random.sample(self.memory, batch_size)
        indexes = (np.random.rand(batch_size) * (len(self.memory) - 1)).astype(int)
        return [self.memory[i] for i in indexes]

    def __len__(self):
        return len(self.memory)


class Env_Runner:

    def __init__(self, env, agent):
        super().__init__()

        self.env = env
        self.agent = agent

        self.logger = Logger("training_info")
        self.logger.log("training_step, return")

        self.ob = self.env.reset()
        self.total_steps = 0

    def run(self, steps):

        obs = []
        actions = []
        rewards = []
        dones = []

        for step in range(steps):

            self.ob = torch.tensor(self.ob)  # uint8
            action = self.agent.e_greedy(
                self.ob.to(device).to(dtype).unsqueeze(0) / 255)  # float32+norm
            action = action.detach().cpu().numpy()

            obs.append(self.ob)
            actions.append(action)

            self.ob, r, done, info, additional_done = self.env.step(action)

            if done:  # real environment reset, other add_dones are for q learning purposes
                self.ob = self.env.reset()
                if "return" in info:
                    self.logger.log(f'{self.total_steps + step},{info["return"]}')

            rewards.append(r)
            dones.append(done or additional_done)

        self.total_steps += steps

        return obs, actions, rewards, dones


def make_transitions(obs, actions, rewards, dones):
    # observations are in uint8 format

    tuples = []

    steps = len(obs) - 1
    for t in range(steps):
        tuples.append((obs[t],
                       actions[t],
                       rewards[t],
                       obs[t + 1],
                       int(not dones[t])))

    return tuples


env_name = 'Jamesbond-v0'

# # hyperparameter
#
num_stacked_frames = 4 #agent history length

replay_memory_size = 250_000 #1_000_000
min_replay_size_to_update = 25_000 #10_000

lr = 25e-5
gamma = 0.99
minibatch_size = 32
steps_rollout = 16

start_eps = 1
final_eps = 0.1

final_eps_frame = 1_000_000

total_steps = 10_000_000

target_net_update = 625  # 10000 steps

# number of steps after which the model is saved
# save_model_steps = 500_000
save_model_steps = 100_000

# init
raw_env = gym.make(env_name)
env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, use_add_done=True)

in_channels = num_stacked_frames
num_actions = env.action_space.n

eps_interval = start_eps - final_eps

agent = Agent(in_channels, num_actions, start_eps).to(device)
target_agent = Agent(in_channels, num_actions, start_eps).to(device)
target_agent.load_state_dict(agent.state_dict())

replay = Experience_Replay(replay_memory_size)
runner = Env_Runner(env, agent)
optimizer = optim.Adam(agent.parameters(), lr=lr)  # optim.RMSprop(agent.parameters(), lr=lr)
huber_loss = torch.nn.SmoothL1Loss()

num_steps = 0
num_model_updates = 0

start_time = time.time()
while num_steps < total_steps:

    # set agent exploration | cap exploration after x timesteps to final epsilon
    new_epsilon = np.maximum(final_eps, start_eps - (eps_interval * num_steps / final_eps_frame))
    agent.set_epsilon(new_epsilon)

    # get data
    obs, actions, rewards, dones = runner.run(steps_rollout)
    transitions = make_transitions(obs, actions, rewards, dones)
    replay.insert(transitions)

    # add
    num_steps += steps_rollout

    # check if update
    if num_steps < min_replay_size_to_update:
        continue

    # update
    for update in range(4):
        optimizer.zero_grad()

        minibatch = replay.get(minibatch_size)

        # uint8 to float32 and normalize to 0-1
        obs = (torch.stack([i[0] for i in minibatch]).to(device).to(dtype)) / 255

        actions = np.stack([i[1] for i in minibatch])
        rewards = torch.tensor([i[2] for i in minibatch]).to(device)

        # uint8 to float32 and normalize to 0-1
        next_obs = (torch.stack([i[3] for i in minibatch]).to(device).to(dtype)) / 255

        dones = torch.tensor([i[4] for i in minibatch]).to(device)

        #  *** double dqn ***
        # prediction

        Qs = agent(torch.cat([obs, next_obs]))
        obs_Q, next_obs_Q = torch.split(Qs, minibatch_size, dim=0)

        obs_Q = obs_Q[range(minibatch_size), actions]

        # target

        next_obs_Q_max = torch.max(next_obs_Q, 1)[1].detach()
        target_Q = target_agent(next_obs)[range(minibatch_size), next_obs_Q_max].detach()

        target = rewards + gamma * target_Q * dones

        # loss
        loss = huber_loss(obs_Q, target)  # torch.mean(torch.pow(obs_Q - target, 2))
        loss.backward()
        optimizer.step()

    num_model_updates += 1

    # update target network
    if num_model_updates % target_net_update == 0:
        target_agent.load_state_dict(agent.state_dict())

    # print time
#    if num_steps % 50000 < steps_rollout:
    if num_steps % 25000 < steps_rollout:
        end_time = time.time()
        print(f'*** total steps: {num_steps} | time(50K): {end_time - start_time} ***')
        start_time = time.time()

    # save the dqn after some time
    if num_steps % save_model_steps < steps_rollout:
        torch.save(agent, f"{env_name}-{num_steps}.pt")

env.close()

# watch

# save agent
torch.save(agent, "agent.pt")
# load agent
agent = torch.load("agent.pt")

# env = gym.make(env_name)
raw_env = gym.make(env_name)
env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)

steps = 5000
ob = env.reset()
agent.set_epsilon(0.025)
agent.eval()
imgs = []
for step in range(steps):
    action = agent.e_greedy(torch.tensor(ob, dtype=dtype).unsqueeze(0).to(device) / 255)
action = action.detach().cpu().numpy()
# action = env.action_space.sample()

env.render()
ob, _, done, info, _ = env.step(action)

time.sleep(0.016)
if done:
    ob = env.reset()
print(info)

imgs.append(ob)

env.close()