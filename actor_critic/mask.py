import numpy as np
import json
import os

def generate_mask(world, agent_num: int, value: int, mask_value: int):
    mask = np.zeros((agent_num, value))
    virtual_intersections = [i for i in world.roadnet["intersections"] if i["virtual"]]
    id2road = world.id2road
    for i in virtual_intersections:
        con_roads = i['roads']  #边缘路段ID
        for r in con_roads:
            mask[id2road[r]] = mask_value
    return mask   

def generate_notrain_id(world):
    idx = []
    virtual_intersections = [i for i in world.roadnet["intersections"] if i["virtual"]]
    id2road = world.id2road 
    for i in virtual_intersections:
        con_roads = i['roads']  #边缘路段ID
        for r in con_roads:
            idx.append(id2road[r])
    idx = np.array(idx)
    train_idx = np.array([value for key, value in world.train_id2road.items()])
    return idx, train_idx

