from actor_critic.OD_agent import OD_agent
from actor_critic.route_attention import Route_Attention
from actor_critic.replay_memory import AgentReplayMemory
import numpy as np
dir_name = 'datasample/4-4/hangzhou_4x4/collect/'
state_batch = np.load(file=dir_name + 'state.npy')
next_state_batch = np.load(file=dir_name + 'next_state.npy')
action_batch = np.load(file=dir_name + 'action.npy')
reward_batch = np.load(file=dir_name + 'reward.npy')

memory = AgentReplayMemory(2000, 3, state_dim=16)
memory.buffer['state'] = state_batch
memory.buffer['next_state'] = next_state_batch
memory.buffer['reward'] = reward_batch
memory.buffer['action'] = action_batch
memory.buffer['done'] = np.zeros((2000, 3, 1))
memory._size = len(state_batch)

import argparse
import gym
import torch.nn.functional as F
from world import World, Route
from torch.utils.tensorboard import SummaryWriter
import os
import json

parser = argparse.ArgumentParser(description="Base Actor-Critic Args")
parser.add_argument(
    "--gamma",
    type=float,
    default=0,
    metavar="G",
    help="discount factor for reward (default: 0.99)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.125,
    metavar="G",
    help="target smoothing coefficient(τ) (default: 0.125)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0,
    metavar="G",
    help="Temperature parameter α determines the relative importance of the entropy\
                    term against the reward (default: 0.2)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    metavar="G",
    help="learning rate (default: 0.001)",
)
parser.add_argument(
    "--batch_size", type=int, default=32, metavar="N", help="batch size (default: 4)"
)
parser.add_argument(
    "--start_episodes", type=int, default=20, metavar="N", help="random sample before"
)
parser.add_argument(
    "--update_after", type=int, default=24, metavar="N", help="update parameters"
)
parser.add_argument(
    "--updates_per_step",
    type=int,
    default=10,
    metavar="N",
    help="model updates per simulator step (default: 1)",
)
parser.add_argument(
    "--target_update_interval",
    type=int,
    default=1,
    metavar="N",
    help="Value target update per no. of updates per step (default: 20)",
)
parser.add_argument(
    "--replay_size",
    type=int,
    default=2000,
    metavar="N",
    help="size of replay buffer (default: 2000)",
)
parser.add_argument(
    "--steps", type=int, default=6, help="number of steps (default: 3600)"
)
parser.add_argument(
    "--thread", type=int, default=8, help="number of threads (default: 8)"
)
parser.add_argument(
    "--num_routes", type=int, default=3, help="number of route choices (default: 3)"
)
parser.add_argument(
    "--action_interval",
    type=int,
    default=1800,
    help="how often agent make decisions (default: 120)",
)
parser.add_argument(
    "--episodes", type=int, default=200, help="training episodes (default: 1)"
)
parser.add_argument(
    "--embed_dim", type=int, default=8, help="training episodes (default: 1)"
)
parser.add_argument(
    "--att_dim", type=int, default=16, help="training episodes (default: 1)"
)
parser.add_argument("--config_file", type=str, help="path of config file")
parser.add_argument("--date", type=str, help="date of running")
args = parser.parse_args()


world = World(args.config_file, thread_num=args.thread, args=args)
action_space = gym.spaces.Box(np.array([-1]), np.array([1]))
route_attention = Route_Attention(world, args)
agent = OD_agent(route_attention, 32, args)

from torch.utils.tensorboard import SummaryWriter
import torch
import os
writer = SummaryWriter('tensorboard/train_from_memory/OD/bs={}'.format(args.batch_size))
dir_name = 'model/train_from_memory/OD//bs={}/'.format(args.batch_size)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
for i in range(1000):
    critic_loss, policy_loss = agent.update_parameters(memory, args.batch_size)
    writer.add_scalar('local_critic_loss', critic_loss, i)
    writer.add_scalar('policy_loss', policy_loss, i)
    print(i,'|critic_loss:', critic_loss, '|policy_loss:', policy_loss)

    if i % 499 == 0 and i > 0:
        torch.save(agent.policy.state_dict(), dir_name + 'policy{}.pth'.format(i+1))
        torch.save(agent.critic.state_dict(), dir_name + 'critic{}.pth'.format(i+1))

