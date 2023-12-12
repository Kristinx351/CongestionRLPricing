import os
import json
import numpy as np
import torch
from torch_geometric.data import Data


def generate_data(edge_index, *args):
    x = list(zip(*args))
    x = [np.concatenate(i) for i in x]
    x = np.array(x)
    x = torch.tensor(x, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    return data


def generate_graph(world, mask=True, *args):
    roads = world.all_roads
    id2road = world.id2road
    non_virtual_inter = world.intersections
    all_intersections = world.all_intersections
    train_id2road = world.train_id2road
    train_id2allroad = world.train_id2allroad
    interinfo = {}
    if mask:
        for inter in non_virtual_inter:
            interinfo[inter.id] = {"asstart": [], "asend": []}
        for road in roads:
            rid = road.id
            if rid in train_id2road:
                startIntersection = road.info["startIntersection"]
                endIntersection = road.info["endIntersection"]
                road_num = train_id2road[rid]
                interinfo[startIntersection]["asstart"].append(road_num)
                interinfo[endIntersection]["asend"].append(road_num) 
    else:
        for inter in all_intersections:
            interinfo[inter["id"]] = {"asstart": [], "asend": []} 
        for road in roads:
            rid = road.id
            startIntersection = road.info["startIntersection"]
            endIntersection = road.info["endIntersection"]
            road_num = id2road[rid]
            interinfo[startIntersection]["asstart"].append(road_num)
            interinfo[endIntersection]["asend"].append(road_num) 
    edge_index = []
    for key, value in interinfo.items():
        for road2 in value["asend"]:
            for road1 in value["asstart"]:
                edge_index.append([road2, road1])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    idx = []
    for trainid in train_id2road:
        idx.append(train_id2allroad[trainid])
    idx = np.array(idx)
    x = list(zip(*args))
    x = [np.concatenate(i) for i in x]
    x = np.array(x)
    if mask:
        x = x[idx]
        x = torch.tensor(x, dtype=torch.float).view(len(train_id2road), -1)
    else:
        x = torch.tensor(x, dtype=torch.float).view(len(roads), -1) 
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    return data


def old_generate_graph(roadnet_config, *args):
    """
    roadnet_config: path of roadnet config
    args:
    state:[[],[],[]]
    action:[[],[],[]]
    """
    path_to_roadnet_input = roadnet_config
    roadnet_json = json.load(open(path_to_roadnet_input, "r"))
    roads = roadnet_json["roads"]
    intersections = roadnet_json["intersections"]
    road_id = []
    for road in roads:
        road_id.append(road["id"])
    road_id.sort()
    id2num = {}
    num2id = {}
    num = 0
    for id in road_id:
        id2num[id] = num
        num2id[num] = id
        num += 1
    interinfo = {}
    for inter in intersections:
        interinfo[inter["id"]] = {"asstart": [], "asend": []}
    for road in roads:
        startIntersection = road["startIntersection"]
        endIntersection = road["endIntersection"]
        road_num = id2num[road["id"]]
        interinfo[startIntersection]["asstart"].append(road_num)
        interinfo[endIntersection]["asend"].append(road_num)

    edge_index = []
    for key, value in interinfo.items():
        for road2 in value["asend"]:
            for road1 in value["asstart"]:
                edge_index.append([road2, road1])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    '''
        x = [[] for index in range(len(roads))]
        for arg in args:
            for index in range(len(roads)):
                x[index].append(arg[index][0])
    '''

    if len(args) == 1:
        x = args
    else:
        x = list(zip(*args))
    x = torch.tensor(x, dtype=torch.float).view(len(roads), -1)
    data = Data(x=x, edge_index=edge_index.t().contiguous())

    return data
