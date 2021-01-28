import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import gym
from model import ActorCritic
from envs import create_jamesbond_env

parser = argparse.ArgumentParser(description='A2C_EVAL')
parser.add_argument('--env-name', default='Jamesbond-v0', metavar='ENV',
                    help='environment to train on (default: Jamesbond-v0)')
parser.add_argument('--num-episodes', type=int, default=20, metavar='NE',
                    help='how many episodes in evaluation (default: 20)')
parser.add_argument('--render', default=False, metavar='R',
                    help='Watch game as it being played')
parser.add_argument('--render-freq', type=int, default=1, metavar='RF',
                    help='Frequency to watch rendered game play')
parser.add_argument('--max-episode-length', type=int, default=100000, metavar='M',
                    help='maximum length of an episode (default: 100000)')
args = parser.parse_args()

env = create_jamesbond_env(args.env_name)

model = ActorCritic(env.observation_space.shape[0], env.action_space)
model.eval()
done = True
env = gym.wrappers.Monitor(env, "{}_monitor".format(args.env_name), force=True)
num_tests = 0
reward_total_sum = 0
for i_episode in range(args.num_episodes):
    state = env.reset()
    episode_length = 0
    reward_sum = 0
    while True:
        if args.render:
            if i_episode % args.render_freq == 0:
                env.render()
        if done:
            model.load_state_dict(torch.load('./models/A2C_Jamesbond_1600_r0_860.pkl'))
            hx = Variable(torch.zeros(1, 256))
        else:
            hx = Variable(hx.data)
        env.render()
        state = torch.from_numpy(state).float()
        value, logit, hx = model(
            (Variable(state.unsqueeze(0)), hx))
        probability = F.softmax(logit, dim=-1)
        action = probability.max(1)[1].data.numpy()
        state, reward, done, _ = env.step(action[0])
        episode_length += 1
        reward_sum += reward
        done = done or episode_length >= args.max_episode_length
        if done:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            print("reward sum: {0}, reward mean: {1:.4f}".format(reward_sum, reward_mean))
            break
            env.close()
