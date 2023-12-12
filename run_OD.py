import argparse
import gym
import torch.nn.functional as F

gym.logger.set_level(40)
import numpy as np
import torch
from actor_critic.OD_agent import OD_agent
from actor_critic.route_attention import Route_Attention
# from torch.utils.tensorboard import SummaryWriter
from actor_critic.replay_memory import AgentReplayMemory
from environment import VCTEnv
from agent.fixedtime_agent import Fixedtime_Agent
from agent.human_eco_agent import HumanEcoAgent
from world import World, Route
from metric import TravelTimeMetric, ThroughputMetric, FuelMetric, TotalCostMetric
from torch.utils.tensorboard import SummaryWriter
import os
import json

def get_focus_id(route_list, world):
    focus_lane = []
    for route in route_list:
        lane_list = world.get_lane_route(route)
        focus_lane.extend(lane_list)
    focus_id = [world.id2lane[lane] for lane in set(focus_lane)]
    return focus_id

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
parser.add_argument(
    "--compare_dim", type=int, default=16, help="training episodes (default: 1)"
)
parser.add_argument("--config_file", type=str, help="path of config file")
parser.add_argument("--date", type=str, help="date of running")
args = parser.parse_args()
config = json.load(open(args.config_file, 'r'))
net = config['dir'].split('/')[1]
flow = config["flowFile"].split('.')[0]
netandflow = net + flow
date = args.date
writer = SummaryWriter('tensorboard/%s/OD/%s/%s/%s' % (date, net, flow, args.batch_size))


world = World(args.config_file, thread_num=args.thread, args=args)

dic_agents = {}

# tsc agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(4)
    agents.append(Fixedtime_Agent(action_space, i.id))
dic_agents["tsc"] = agents


memory = AgentReplayMemory(args.replay_size, agent_num=3, state_dim=args.att_dim, attention=True)
# cp agents
agents = []
action_space = gym.spaces.Box(np.array([-1]), np.array([1]))
route_attention = Route_Attention(world, args)
OD_pair = {0: "road_2_1_1-road_2_16_1", 1: "road_1_1_1-road_3_15_0"}
# OD_pair = {0: "road_1_1_0-road_4_4_0"}
world_route = {key:[] for key in OD_pair}
for key in OD_pair:
    for i in range(3):
        road = OD_pair[key].split("-")
        route_item = Route(world, road[0], road[-1], "_".join([str(key), str(i)]))
        world_route[key].append(route_item)

dic_agents["cp"] = OD_agent(route_attention, args.compare_dim, args)

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


# create env
env = VCTEnv(world, dic_agents, metric, args)
# 16x3
train_movement = [54, 55, 58, 66, 70, 74, 82, 100, 112, 124, 142, 154, 166, 184, 196, 208, 226, 238, 250, 268, 280, 292, 310, 322, 334, 352, 364, 376, 394, 406, 418, 436, 448, 460, 478, 490, 502, 520, 532, 544, 562, 563, 574, 575, 586, 587, 601, 605, 613, 616, 627, 642, 674]
# 4x4
# train_movement = [86, 193, 98, 16, 123, 70, 94, 147, 160, 15, 85, 73, 106, 206, 214, 181]
train_id = np.array(train_movement)

def train(args, metric_name):
    loss = 0
    detail = {}
    raw_state = []
    raw_next_state = []
    for e in range(args.episodes):
        if e == 1 or e == 199:
            detail[e] = {}
        interval_reward_record = []
        state_record = []
        action_record = []
        reward_record = []
        episode_reward = 0
        travel_time_record = []
        throughput_record = []
        done = False
        env.reset()  # 仅关心road的state
        world_state = world.get_delta_toll()
        state = []
        for key, OD in world_route.items():
            alter_route = [route.lane_list for route in OD]
            OD_state = dic_agents['cp'].get_obs(world_state, alter_route)
            # raw_state.append(raw)
            state.append(OD_state)
        for key, route_type in world_route.items():
            for singe_route in route_type:
                singe_route.reset()
        print('route', " |episodes is : ", e)
        reward_list = []
        dic_actions = {}  # e < 30 set_seed
        pre_pass_distance = np.zeros((len(world.all_lanes), 1))

        for i in range(args.steps):
            print('route', "|", i, "/", args.steps, " |episodes is : ", e)
            key = "cp"
            dic_actions[key] = []  # 所有路段的动作
            route_action = [] #保存所有OD action
            if np.random.rand() > world.epsilon:
                for od_id, OD in world_route.items(): #od_id对应route编号
                    # alter_route = [route.lane_list for route in OD]
                    OD_action = dic_agents["cp"].get_action(state[od_id], False) #shape:(1, 3)
                    route_action.append(OD_action)
                    for idx, singe_route in enumerate(OD):
                        singe_route.step(OD_action[idx])
            else:
                world.epsilon = world.epsilon * world.epsilon_decay
                print("Exploration")
                for od_id, OD in world_route.items():
                    OD_action = -1 + 2*np.random.random((3,1))
                    route_action.append(OD_action)
                    for idx, singe_route in enumerate(OD):
                        singe_route.step(OD_action[idx])

            for t in range(args.action_interval):
                # traffic light take action every second
                key = "tsc"
                dic_actions[key] = []
                for id, agent in enumerate(dic_agents[key]):
                    dic_actions[key].append(agent.get_action(world))
                
                next_state, reward, done, info, vehicle = env.step(dic_actions)
                dic_actions["vehicle"] = []
                for id, agent in enumerate(dic_agents["vehicle"]):
                    if agent is not None and agent.vehicle.id in info and agent.vehicle.monitor:
                        key_id = int(agent.vehicle.id.split("_")[1])
                        route_choice = world_route[key_id]
                        price_list = [wr.price for wr in route_choice]
                        cost = - torch.FloatTensor(price_list)
                        pro = F.softmax(cost, dim=0)
                        choice = np.random.choice(3,1,p=np.array(pro))[0]
                        # chosen_route_idx = price_list.index(min(price_list))
                        chosen_action = route_choice[choice].road_list
                        dic_actions["vehicle"].append(chosen_action)
                        world.vehicle_route[str(key_id)][choice] += 1
                    else:
                        dic_actions["vehicle"].append([])
                reward_list.append(reward)
                if e == 1 or e == 199:
                    detail[e][1800 * i + t] = vehicle
                for ind_m in range(len(env.metric)):
                    env.metric[ind_m].update(done=False)

            pass_distance = np.mean(reward_list, axis=0)
            next_state = []
            world_state = world.get_delta_toll()
            for key, OD in world_route.items():
                alter_route = [route.lane_list for route in OD]
                OD_state = dic_agents['cp'].get_obs(world_state, alter_route)
                # if i != 0:
                #     raw_next_state.append(raw) 
                next_state.append(OD_state)
                # if i != (args.steps - 1):
                #     raw_state.append(raw) #最后一个一个是多余的
            if i != 0:
                # rewards = pass_distance - pre_pass_distance
                rewards = pass_distance
                episode_reward += np.sum(rewards[train_id])
                interval_reward_record.append(np.sum(rewards[train_id]))
                reward_list = []
                route_reward = []
                for key, route_type in world_route.items():
                    alter_route = [route.lane_list for route in OD]
                    route_reward.append(dic_agents["cp"].get_reward(rewards, alter_route))
                for id, agent in enumerate(next_state):
                    memory.push(
                        state=state[id],
                        action=route_action[id],
                        reward=route_reward[id],
                        next_state=next_state[id],
                        done=done[id],
                    )
                state_record.append([s.detach().numpy() for s in state])
                # print(state)
                action_record.append(route_action)
                reward_record.append(rewards)
                travel_time_record.append(env.metric[0].update(done=False))
                throughput_record.append(env.metric[1].update(done=False))
            state = next_state
            # pre_pass_distance = pass_distance

        # the following code is done for each episode
        if len(memory) > args.batch_size and e % 10 == 0:
            for j in range(args.updates_per_step):
                critic_loss, policy_loss = dic_agents["cp"].update_parameters(memory, args.batch_size)
                writer.add_scalar('critic_loss', critic_loss, loss)
                writer.add_scalar('policy_loss', policy_loss, loss)
                loss += 1
        if e % 30 == 0:
            dir_name = 'model/%s/%s/%s/OD/%s/%s/' % (date, net, flow, args.batch_size, e)
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            torch.save(dic_agents["cp"].policy.state_dict(), dir_name + 'policy.pth')
            torch.save(dic_agents["cp"].critic.state_dict(), dir_name + 'critic.pth')
            torch.save(dic_agents["cp"].route_attention.attention.state_dict(), dir_name + 'attention.pth')
        
        dir_name = 'train_log/%s/%s/%s/OD/%s/%s/' % (date, net, flow, args.batch_size, e)
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
        if e == 1 or e == 199:
            vehicle_pass = json.dumps(detail, indent=2)
            with open(dir_name + 'vehicle_pass.json', 'w') as json_file:
                json_file.write(vehicle_pass)
        
        reward_json = {}
        reward_json['interval_reward'] = interval_reward_record
        reward_str = json.dumps(reward_json, indent=2)
        with open(dir_name + 'interval_reward.json', 'w') as json_file:
            json_file.write(reward_str)
        
        if e % 20 == 0:
            dir = 'datasample/%s/%s/%s/OD/%s/' % (date, net, flow, e)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            buffer_size = memory._size
            np.save(dir + 'action.npy', memory.buffer['action'][:buffer_size])
            torch.save(memory.buffer['state'][:buffer_size], dir + 'state.npy')
            torch.save(memory.buffer['next_state'][:buffer_size], dir + 'next_state.npy')
            np.save(dir + 'reward.npy', memory.buffer['reward'][:buffer_size])
            # np.save(dir + 'raw_state.npy', np.array(raw_state))
            # np.save(dir + 'raw_next_state.npy', np.array(raw_next_state))




if __name__ == "__main__":
    train(args, metric_name)
