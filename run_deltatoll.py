import argparse
from time import time
import os
import gym
import time
import logging

from agent.delta_agent import Price_Agent
from actor_critic.replay_memory import AgentReplayMemory
gym.logger.set_level(40)
import numpy as np
import itertools
import torch
from agent.base import BaseAgent
from agent.human_eco_agent import HumanEcoAgent
# from torch.utils.tensorboard import SummaryWriter
from environment import VCTEnv, VehiclEnv, CPEnv, TSCEnv
# TODO: change with the roadnet
# porto
from agent.fixedtime_agent2 import Fixedtime_Agent
# others
# from agent.fixedtime_agent import Fixedtime_Agent
from agent.charge_agent import Charge_Agent
from world import World
from metric import TravelTimeMetric, ThroughputMetric, FuelMetric, TotalCostMetric, throughput, travel_time
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Formula Price Args")
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
    "--replay_size",
    type=int,
    default=2000,
    metavar="N",
    help="size of replay buffer (default: 2000)",
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
parser.add_argument(
    "--batch_size", type=int, default=32, metavar="N", help="batch size (default: 32)"
)
# TODO: change with the roadnet
parser.add_argument("--config_file", type=str, default="dataset/porto/config.json", help="path of config file")
args = parser.parse_args()
'''
    args=[
        "--thread",
        "8",
        "--roadnet_file",
        "dataset/hangzhou_4x4/roadnet.json",
    ]
'''
config = json.load(open(args.config_file, 'r'))
net = config['dir'].split('/')[1]
flow = config["flowFile"].split('.')[0]
netandflow = net + flow

world = World(args.config_file, thread_num=args.thread, args=args)

dic_agents = {}


# TODO: change with the roadnet
# porto tsc agents
agents = []
action_space = gym.spaces.Discrete(4)
for i in world.intersections:
    # print("%s has %d outroads." % (i.id, len(i.out_roads)))
    agents.append(Fixedtime_Agent(len(i.out_roads), action_space, i.id))
dic_agents["tsc"] = agents

# tsc agents
# agents = []
# for i in world.intersections:
#     action_space = gym.spaces.Discrete(4)
#     agents.append(Fixedtime_Agent(action_space, i.id))
# dic_agents["tsc"] = agents

# # cp agents
# agents = []
# action_space = gym.spaces.Box(np.array([0]), np.array([10]))
# for i in world.all_lanes:
#     agents.append(Price_Agent(i.id, world, R, beta))
# dic_agents['cp'] = agents

# vehicle agents
agents = []
vehicle_action_space = gym.spaces.Discrete(args.num_routes)
for i in world.vehicles:
    agents.append(HumanEcoAgent(i, world))
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

# Memory
memory = AgentReplayMemory(args.replay_size, agent_num=len(world.all_lanes), state_dim=1)

# Agents num:
a = 4
b = 4
a_num = 6*(a*(b+1) + b*(a+1))

# # create env
# env = VCTEnv(world, dic_agents, metric, args)


def test(args, metric_name, R, beta, round_id):
    # cp agents
    agents = []
    action_space = gym.spaces.Box(np.array([0]), np.array([1]))
    for i in world.all_lanes:
        agents.append(Price_Agent(i, world, R, beta))
    dic_agents['cp'] = agents

    # create env
    env = VCTEnv(world, dic_agents, metric, args)
    # TODO: change with roadnet:
    # 16x3
    # train_movement = [54, 55, 58, 66, 70, 74, 82, 100, 112, 124, 142, 154, 166, 184, 196, 208, 226, 238, 250, 268, 280, 292, 310, 322, 334, 352, 364, 376, 394, 406, 418, 436, 448, 460, 478, 490, 502, 520, 532, 544, 562, 563, 574, 575, 586, 587, 601, 605, 613, 616, 627, 642, 674]
    # 4x4
    # train_movement = [86, 193, 98, 16, 123, 70, 94, 147, 160, 15, 85, 73, 106, 206, 214, 181]
    # porto
    train_movement = [292, 273, 185, 475, 54, 451, 478, 574, 119, 69, 319, 383, 499, 116, 222, 402, 537, 484, 288, 338,
                      88, 562, 109, 372, 407, 152, 387, 438, 458, 79, 364, 347, 210, 245, 540, 39, 193, 411, 299, 175,
                      162, 394, 369, 40, 379, 414, 65, 509, 491, 238, 274, 337, 106, 204, 443, 52, 276, 445, 466, 544,
                      424, 352, 373, 542, 286, 435, 510, 82, 94, 550, 13, 498, 310, 505, 137, 351, 67, 565, 280, 418,
                      586, 457, 530, 431, 511, 77, 340, 143, 159, 70, 183, 100, 172, 548, 397, 64, 577, 253, 551, 84,
                      141, 217, 376, 476, 391, 541, 208, 265, 266, 233]
    train_id = np.array(train_movement)
    # record
    initial_price = 5
    interval_reward_record = []
    detail = {}
    for e in range(args.episodes):
        detail[e] = {}
        state_record = []
        action_record = []
        reward_record = []
        travel_time_record = []
        throughput_record = []
        done = False
        state = env.reset()  # road state only 
        env.eng.set_save_replay(True)
        env.eng.set_replay_file("delta_toll_%s.txt" % (e))
        print("delta_toll", " |episodes is : ", e)
        reward_list = []
        dic_actions = {}
        for i in range(args.steps):
            # road & vehicle take action only if the time is the 'interval'
            if i == 2:
               env.eng.set_save_replay(False)
            print("delta_toll", "|", i, "/", args.steps)
            key = "cp"
            dic_actions[key] = []  # Action of all roads

            if i == 0:
                # dic_actions[key] = [agent.get_action(5, i, id) for id, agent in enumerate(dic_agents['cp'])]
                dic_actions[key] = np.array([[initial_price]*1]*a_num)
                # print(dic_actions[key])
            else:
                for id, agent in enumerate(dic_agents['cp']):
                    if id in train_id:
                        dic_actions[key].append(agent.get_action(i, id))
                    else:
                        dic_actions[key].append(np.array([5]))
                dic_actions[key] = np.array(dic_actions[key])

            for t in range(args.action_interval):
                # print("formula", "|", t, "/", args.action_interval)
                # traffic light take action every second
                key = "tsc"
                dic_actions[key] = []
                # for id, agent in enumerate(dic_agents[key]):
                #     dic_actions[key].append(agent.get_action(world))
                dic_actions[key] = [agent.get_action(world) for agent in dic_agents[key]]

                """
                env.step
                <<<
                """
                next_state, reward, done, info, vehicle = env.step(dic_actions)
                # print("next1", next_state)
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
                for id, agent in enumerate(dic_agents['cp']):
                    if id in train_id:
                        agent.update()
            next_state = env.dic_env["cp"].get_state()
            # print(len(next_state[0]))
            pass_distance = np.mean(reward_list, axis=0)
            if i != 0:
                # rewards = pass_distance - pre_pass_distance
                rewards = pass_distance
                interval_reward_record.append(np.sum(rewards[train_id]))
                reward_list = []
                # print(len(state),len(dic_actions["cp"]), len(next_state), len(done))
                # print(state)
                # print(len(state[0]))
                memory.push(
                    state=state,
                    action=dic_actions["cp"],
                    reward=rewards,
                    next_state=next_state,
                    done=done,
                )
                state_record.append(state)
                action_record.append((dic_actions["cp"] + 1) * 5)
                reward_record.append(rewards)
                travel_time_record.append(env.metric[0].update(done=False))
                throughput_record.append(env.metric[1].update(done=False))
            # pre_pass_distance = pass_distance

        # the following code is done for each episode
        dir_name = 'train_log/5-8/%s/%s/%s/delta_tolling/%s/' % (round_id, net, flow, e)
        # 0-d1; 1-d2
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        state_record = np.concatenate(state_record)
        action_record = np.concatenate(action_record)
        reward_record = np.concatenate(reward_record)
        travel_time_record = np.array(travel_time_record)
        TT_detail = env.metric[0].update(done=True)
        record = {'state': state_record.tolist(), 'action': action_record.tolist(), 'reward': reward_record.tolist(),
                'TT': travel_time_record.tolist(), 'throughput': throughput_record}
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
        dir = 'datasample/5-8/%s/%s/deltatoll/' % (net, flow)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        buffer_size = memory._size
        np.save(dir + 'state.npy', memory.buffer['state'][:buffer_size])
        np.save(dir + 'action.npy', memory.buffer['action'][:buffer_size])
        np.save(dir + 'reward.npy', memory.buffer['reward'][:buffer_size])
        np.save(dir + 'next_state.npy', memory.buffer['next_state'][:buffer_size])


if __name__ == "__main__":
    # time0=time.time()
    test(args, metric_name, R=10e-1, beta=8, round_id=12)
