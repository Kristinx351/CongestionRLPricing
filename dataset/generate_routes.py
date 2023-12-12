import os
import json
from typing import final

import sys

sys.path.append("../")
from world import World
import argparse
import json

parser = argparse.ArgumentParser(description="Run Example")
parser.add_argument("--config_file", type=str, help="path of config file")
parser.add_argument(
    "--agents_mode",
    default="tsc:default,vehicle:default,cp:default",
    type=str,
    help="tsc:default,vehicle:default,cp:default",
)
parser.add_argument("--memo", type=str, default="default")
parser.add_argument("--thread", type=int, default=1, help="number of threads")
parser.add_argument("--steps", type=int, default=3600, help="number of steps")
parser.add_argument("--num_routes", type=int, default=3, help="number of route choices")
parser.add_argument(
    "--action_interval", type=int, default=20, help="how often agent make decisions"
)
parser.add_argument("--episodes", type=int, default=6, help="training episodes")
parser.add_argument("--save_model", action="store_true", default=False)
parser.add_argument("--load_model", action="store_true", default=False)
parser.add_argument(
    "--save_rate",
    type=int,
    default=20,
    help="save model once every time this many episodes are completed",
)
args = parser.parse_args(
    args=["--config_file", "manhattan_28x7/config.json"]
)
print(args)

print(os.getcwd())
# Dijkstra
world = World(args.config_file, thread_num=args.thread, args=args)
MAX = 1e6


# load roadnet
path_to_roadnet_input = os.path.join(
    "manhattan_28x7", "roadnet_28_7.json"
)
path_to_route_output = os.path.join(
    "manhattan_28x7", "optional_routes_optimized.json"
)
roadnet_json = json.load(open(path_to_roadnet_input, "r"))
roads = roadnet_json["roads"]
intersections = roadnet_json["intersections"]

# some prepared work
inter2road = {}  # 以inter开始的road
road2inter = {}  # 每个road结束的inter

not_start = []
not_end = []
#边缘路段不会作为起始路段
inter_virtual = {}  #虚拟路口，仅与一条road相连即路网边缘
for inter in intersections:
    inter_virtual[inter["id"]] = inter["virtual"]

for road in roads:
    road2inter[road["id"]] = road["endIntersection"]
    start_inter = road["startIntersection"]
    if start_inter not in inter2road:
        inter2road[start_inter] = []
    inter2road[start_inter].append(road["id"])
    if inter_virtual[road['startIntersection']]:
        not_end.append(road['id'])
        not_start.append(road['id'])
    if inter_virtual[road['endIntersection']]:
        not_start.append(road['id'])





forbiddenr2r = {}  # 禁止连续行驶的road（由cityflow不可掉头的性质得出）
# 'id': 'road_0_1_0'
for road in roads:
    rid = road["id"]
    rlen = len(rid)
    tail = rid[-1]
    x = int(rid[5])
    if rlen == 10:
        y = int(rid[-3])
    else:
        y = int(rid[7:9])
    if tail == "0":
        forbiddenr2r[rid] = "road_{}_{}_2".format(x + 1, y)
    elif tail == "1":
        forbiddenr2r[rid] = "road_{}_{}_3".format(x, y + 1)
    elif tail == "2":
        forbiddenr2r[rid] = "road_{}_{}_0".format(x - 1, y)
    else:
        forbiddenr2r[rid] = "road_{}_{}_1".format(x, y - 1)

with open(
    "../dataset/manhattan_16x3/anon_16_3_newyork_real.json", "r", encoding="utf8"
) as fp:
    json_data = json.load(fp)
#     print('这是文件中的json数据：',json_data)
route_vehicle_num = {}
for vehicle in json_data:
    route = vehicle["route"]
    for road in route:
        if road not in route_vehicle_num:
            route_vehicle_num[road] = 0
        route_vehicle_num[road] = route_vehicle_num[road] + 1


# functions
def find_route_BFS(start, end):
    # 最小跳数的路径
    # print("find_route_BFS")
    list_routes = [[start]]
    list_new_routes = []
    final_route = []
    flag = False
    BFS_route = []
    num = 0
    for i in range(20):
        if len(list_routes) != 0:
            for ind_route, route in enumerate(list_routes):
                this_start = route[-1]  # dequeue this_start
                #                 print("this_start:",this_start)
                if this_start != end:
                    next_inter = road2inter[this_start]  # neighbor point(inter)
                    if next_inter in inter2road:
                        for next_road in inter2road[next_inter]:
                            if (next_road not in route) and (
                                next_road != forbiddenr2r[this_start]
                            ):
                                list_new_routes.append(route + [next_road])
                else:
                    num += 1
                    final_route.append(route)
                    # list_new_routes.append(route)
            list_routes = list_new_routes[:]
            if num == 3:
                break
            del list_new_routes
            list_new_routes = []
    #             print(list_routes)

    for ind_route, route in enumerate(final_route):
        if route[-1] == end:
            BFS_route.append(route)
            #             print("{}-{}:{}".format(start,end,route))
            #             print(len(route))
            flag = True

    if flag == False:
        # print("None")
        return None
    if len(BFS_route) > 3:
        BFS_route.sort(key=lambda i: len(i))
        BFS_route = BFS_route[:3]
    return BFS_route


def find_route_Dij(start, end):
    #     points = self.world.all_roads[self.world.id2road[start]].info["points"]
    #     start_length = pow(pow((points[0]['x'] - points[1]['x']), 2) + pow((points[0]['y'] - points[1]['y']), 2), 0.5)
    # 最短路径
    # print("find_route_Dij")
    dist = {}
    path = {}  # prior point
    edge = {}
    # 初始化
    for road in roads:
        id = road["id"]
        path[id] = -1
        edge[id] = MAX
        dist[id] = edge[id]
    next_inter = road2inter[start]  # neighbor point(inter)
    if next_inter in inter2road:
        for next_road in inter2road[next_inter]:
            if next_road != forbiddenr2r[start]:
                points = world.all_roads[world.id2road[next_road]].info["points"]
                length = pow(
                    pow((points[0]["x"] - points[1]["x"]), 2)
                    + pow((points[0]["y"] - points[1]["y"]), 2),
                    0.5,
                )
                path[next_road] = start
                edge[next_road] = length
                dist[next_road] = length
    result = [start]  # 存放结果集

    for i in range(230):
        min = MAX
        # 每次寻找一个最小值，放入结果集
        for road in roads:
            id = road["id"]
            if (id not in result) and (dist[id] < min):
                min = dist[id]
                u = id
        if min == MAX:
            # print("None")
            return None

        result.append(u)
        if u == end:
            break
        # update distance
        next_inter = road2inter[u]  # neighbor point(inter)
        if next_inter in inter2road:
            for next_road in inter2road[next_inter]:
                points = world.all_roads[world.id2road[next_road]].info["points"]
                length = pow(
                    pow((points[0]["x"] - points[1]["x"]), 2)
                    + pow((points[0]["y"] - points[1]["y"]), 2),
                    0.5,
                )
                if (
                    (next_road not in result)
                    and ((dist[u] + length) < dist[next_road])
                    and (next_road != forbiddenr2r[u])
                ):
                    dist[next_road] = dist[u] + length
                    path[next_road] = u

    route = [end]

    if end not in result:
        # print("None")
        return None

    while route[-1] != start:
        r = route[-1]
        route.append(path[r])
    route.reverse()

    return route


def find_route_Time(start, end):
    #     points = self.world.all_roads[self.world.id2road[start]].info["points"]
    #     start_length = pow(pow((points[0]['x'] - points[1]['x']), 2) + pow((points[0]['y'] - points[1]['y']), 2), 0.5)
    # 结合车辆和路程的最少时间
    # print("find_route_Time")
    dist = {}
    path = {}
    edge = {}
    # 初始化
    for road in roads:
        id = road["id"]
        path[id] = -1
        edge[id] = MAX
        dist[id] = edge[id]
    next_inter = road2inter[start]  # neighbor point(inter)
    if next_inter in inter2road:
        for next_road in inter2road[next_inter]:
            if next_road != forbiddenr2r[start]:
                points = world.all_roads[world.id2road[next_road]].info["points"]
                length = pow(
                    pow((points[0]["x"] - points[1]["x"]), 2)
                    + pow((points[0]["y"] - points[1]["y"]), 2),
                    0.5,
                )
                if next_road in route_vehicle_num:
                    length += route_vehicle_num[next_road]
                path[next_road] = start
                edge[next_road] = length
                dist[next_road] = length
    result = [start]  # 存放结果集
    for i in range(230):
        min = MAX
        # 寻找最小值
        for road in roads:
            id = road["id"]
            if (id not in result) and (dist[id] < min):
                min = dist[id]
                u = id
        if min == MAX:
            print("None")
            return None

        result.append(u)
        if u == end:
            break
        # update distance
        next_inter = road2inter[u]  # neighbor point(inter)
        if next_inter in inter2road:
            for next_road in inter2road[next_inter]:
                points = world.all_roads[world.id2road[next_road]].info["points"]
                length = pow(
                    pow((points[0]["x"] - points[1]["x"]), 2)
                    + pow((points[0]["y"] - points[1]["y"]), 2),
                    0.5,
                )
                if next_road in route_vehicle_num:
                    length += route_vehicle_num[next_road]
                if (
                    (next_road not in result)
                    and ((dist[u] + length) < dist[next_road])
                    and (next_road != forbiddenr2r[u])
                ):
                    dist[next_road] = dist[u] + length
                    path[next_road] = u

    route = [end]

    if end not in result:
        # print("None")
        return None

    while route[-1] != start:
        r = route[-1]
        route.append(path[r])
    route.reverse()
    return route

    # 具体导出最小跳数


BFS_routes = {}
for inter in intersections:
    inter_id = inter["id"]
    for start_road in inter2road[inter_id]:
        if start_road in not_start:
            continue
        for end_road in roads:
            end_road_id = end_road["id"]
            if start_road == end_road_id or end_road_id in not_end:
                continue
            BFS_route = find_route_BFS(start_road, end_road_id)
            BFS_routes[start_road + "-" + end_road_id] = BFS_route
            print(start_road, "-", end_road_id, ": ", BFS_route)

Dijkstra_routes = {}
for inter in intersections:
    inter_id = inter["id"]
    for start_road in inter2road[inter_id]:
        if start_road in not_start:
            continue
        for end_road in roads:
            end_road_id = end_road["id"]
            if start_road == end_road_id  or end_road_id in not_end:
                continue
            D_route = find_route_Dij(start_road, end_road_id)
            Dijkstra_routes[start_road + "-" + end_road_id] = D_route
            print(start_road, "-", end_road_id, ": ", D_route)

Time_routes = {}
for inter in intersections:
    inter_id = inter["id"]
    for start_road in inter2road[inter_id]:
        if start_road in not_start:
            continue
        for end_road in roads:
            end_road_id = end_road["id"]
            if start_road == end_road_id or end_road_id in not_end:
                continue
            T_route = find_route_Time(start_road, end_road_id)
            Time_routes[start_road + "-" + end_road_id] = T_route
            print(start_road, "-", end_road_id, ": ", T_route)


route_choices = {}
for BFS in BFS_routes:
    if (BFS not in route_choices) and (BFS_routes[BFS] != None):
        route_choices[BFS] = BFS_routes[BFS]

for D in route_choices:
    if Dijkstra_routes[D] not in route_choices[D]:
        route_choices[D].insert(1, Dijkstra_routes[D])

for T in route_choices:
    if Time_routes[T] not in route_choices[T]:
        route_choices[T].insert(1, Time_routes[T])

# 取前三条
for route in route_choices:
    route_choices[route] = route_choices[route][:3]
# print(route_choices["road_4_4_2-road_1_1_2"])

# 导出
json.dump(route_choices, open(path_to_route_output, "w"), indent=2)
