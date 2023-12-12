import argparse
import datetime
import imp
import gym

gym.logger.set_level(40)
import numpy as np
import itertools
import torch
from actor_critic.BaseAC import BaseAC

# from torch.utils.tensorboard import SummaryWriter
from actor_critic.replay_memory import AgentReplayMemory
from environment import VCTEnv, VehiclEnv, CPEnv, TSCEnv
from agent.fixedtime_agent import Fixedtime_Agent
from agent.random_human_agent import Random_Human
from world import World
from metric import TravelTimeMetric, ThroughputMetric, FuelMetric, TotalCostMetric
import os
import json
parser = argparse.ArgumentParser(description="Base Actor-Critic Args")
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
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
    default=0.2,
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
    "--batch_size", type=int, default=100, metavar="N", help="batch size (default: 4)"
)
parser.add_argument(
    "--start_episodes", type=int, default=10, metavar="N", help="random sample before"
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
    default=200000,
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
    "--episodes", type=int, default=1, help="training episodes (default: 1)"
)
parser.add_argument("--config_file", type=str, help="path of config file")
args = parser.parse_args()
config = json.load(open(args.config_file, 'r'))
net = config['dir'].split('/')[1]
flow = config["flowFile"].split('.')[0]
netandflow = net + flow


world = World(args.config_file, thread_num=args.thread, args=args)

dic_agents = {}

# tsc agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(4)
    agents.append(Fixedtime_Agent(action_space, i.id))
dic_agents["tsc"] = agents

agent_action_space = gym.spaces.Box(np.array([-1]), np.array([1]))
dic_agents["cp"] = BaseAC(agent_action_space, args)  # local share

# vehicle agents
agents = []
vehicle_action_space = gym.spaces.Discrete(args.num_routes)
for i in world.vehicles:
    agents.append(Random_Human(i, world))
dic_agents["vehicle"] = agents

# create metric
metric = [
    TravelTimeMetric(world),
    ThroughputMetric(world),
    FuelMetric(world),
    TotalCostMetric(world),
]
metric_name = [
    "Average Travel Time",
    "Average throughput",
    "Average fuel cost",
    "Average total cost",
]


# create env
env = VCTEnv(world, dic_agents, metric, args)
# 16x3
# train_movement = [54, 55, 58, 66, 70, 74, 82, 100, 112, 124, 142, 154, 166, 184, 196, 208, 226, 238, 250, 268, 280, 292, 310, 322, 334, 352, 364, 376, 394, 406, 418, 436, 448, 460, 478, 490, 502, 520, 532, 544, 562, 563, 574, 575, 586, 587, 601, 605, 613, 616, 627, 642, 674]
# 4x4
train_movement = [86, 193, 98, 16, 123, 70, 94, 147, 160, 15, 85, 73, 106, 206, 214, 181]
train_id = np.array(train_movement)
def train(args, metric_name):
    detail = {}
    interval_reward_record = []
    for e in range(args.episodes):
        detail[e] = {}
        reward_record = []
        episode_reward = 0
        travel_time_record = []
        throughput_record = []
        done = False
        state = env.reset()  # 仅关心road的state
        print('random', " |episodes is : ", e)
        reward_list = []
        dic_actions = {} 
        pre_pass_distance = np.zeros((len(world.all_lanes), 1))

        for i in range(args.steps):
            if i == 4:
                env.eng.set_save_replay(True)
                env.eng.set_replay_file("random_%s.txt" % e)
            key = "cp"
            # 所有路段的动作
            dic_actions[key] = np.zeros((len(world.all_lanes), 1))

            for t in range(args.action_interval):
                # traffic light take action every second
                key = "tsc"
                dic_actions[key] = [agent.get_action(
                    world) for id, agent in enumerate(dic_agents[key])]
                
                next_state, reward, done, info, vehicle = env.step(dic_actions)
                reward_list.append(reward)
                detail[e][1800 * i + t] = vehicle
                dic_actions["vehicle"] = []
                for id, agent in enumerate(dic_agents["vehicle"]):
                    if agent is not None and agent.vehicle.id in info and agent.vehicle.monitor:
                        dic_actions["vehicle"].append(agent.get_action(world))
                    else:
                        dic_actions["vehicle"].append([])
                reward_list.append(reward)
                for ind_m in range(len(env.metric)):
                    env.metric[ind_m].update(done=False)
            pass_distance = np.mean(reward_list, axis=0)
            if i != 0:
                # rewards = pass_distance - pre_pass_distance
                rewards = pass_distance
                interval_reward_record.append(np.sum(rewards[train_id]))
                reward_list = []
                reward_record.append(rewards)
                travel_time_record.append(env.metric[0].update(done=False))
                throughput_record.append(env.metric[1].update(done=False))
            # pre_pass_distance = pass_distance

        dir_name = 'train_log/2-20/%s/%s/random/%s/' % (net, flow, e)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        reward_record = np.concatenate(reward_record)
        travel_time_record = np.array(travel_time_record)
        TT_detail = env.metric[0].update(done=True)
        record = {'reward': reward_record.tolist(),'TT': travel_time_record.tolist(), 'throughput': throughput_record}
        json_str = json.dumps(record, indent=2)
        with open(dir_name + 's-a-r-t.json', 'w') as json_file:
            json_file.write(json_str)
        TT_str = json.dumps(TT_detail, indent=2)
        with open(dir_name + 'TT_detail.json', 'w') as json_file:
            json_file.write(TT_str)
        reroute = json.dumps(world.vehicle_route, indent=2)
        with open(dir_name + 'reroute.json', 'w') as json_file:
            json_file.write(reroute)
        vehicle_pass = json.dumps(detail, indent=2)
        with open(dir_name + 'vehicle_pass.json', 'w') as json_file:
            json_file.write(vehicle_pass)
        
        reward_json = {}
        reward_json['interval_reward'] = interval_reward_record
        reward_str = json.dumps(reward_json, indent=2)
        with open(dir_name + 'interval_reward.json', 'w') as json_file:
            json_file.write(reward_str)
        




if __name__ == "__main__":
    train(args, metric_name)
