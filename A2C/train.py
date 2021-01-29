import math
import os
import sys
import csv

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable


def train(args, model, env, optimizer=None, *, iteration=0):
    torch.manual_seed(args.seed)

    # env = create_atari_env(args.env_name)
    # env = create_car_racing_env()
    print("env: ", env.observation_space.shape, env.action_space)
    # env.seed(args.seed)
    file = open('training_info.csv', 'a+', newline='')
    writer = csv.writer(file)
    # model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    # print ("state: ", state.shape)
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    sum_reward = 0
    # u = 0
    # while u < args.num_updates:
    for u in tqdm(torch.arange(args.num_updates)):
        #print ("update: ", u)
        episode_length += 1
        # Sync with the shared model
        # model.load_state_dict(shared_model.state_dict())
        if done:
            hx = Variable(torch.zeros(1, 512))
        else:
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            value, logit, hx = model(
                (Variable(state.unsqueeze(0)), hx))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = torch.multinomial(prob, 1)
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done, info = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            # if ('ale.lives' in info):
            #     reward -= (lives > info['ale.lives']) * 10 # punishment for dying
            #     lives = info['ale.lives']
            sum_reward += reward

            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                writer.writerow([iteration*args.num_updates+u.item(), sum_reward])
                sum_reward = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), hx))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        optimizer.step()
        # u += 1
    file.close()
