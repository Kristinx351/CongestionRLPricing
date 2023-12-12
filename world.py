import json
import os.path as osp
from xml.etree.ElementTree import tostring
import cityflow
from torch_geometric.data import Data
import numpy as np
from math import atan2, pi
import torch
import os
from time import time

def _get_direction(road, out=True):
    if out:
        x = road["points"][1]["x"] - road["points"][0]["x"]
        y = road["points"][1]["y"] - road["points"][0]["y"]
    else:
        x = road["points"][-2]["x"] - road["points"][-1]["x"]
        y = road["points"][-2]["y"] - road["points"][-1]["y"]
    tmp = atan2(x, y)
    return tmp if tmp >= 0 else (tmp + 2 * pi)


class Vehicle(object):
    def __init__(self, vehicle_id, vehicle, world):
        self.id = vehicle_id
        self.eng = world.eng
        self.info = vehicle
        self.in_system = None
        self.change = False
        self.original_route = vehicle["route"][:-1].split(" ")
        self.last_distance = float(self.eng.get_vehicle_info(self.id)["distance"])
        self.route = self.original_route[1:]  # 记录所选路线
        try:
            self.route_choices = world.route[
                "-".join([self.original_route[1], self.original_route[-1]])
            ]
        except KeyError:
            if self.original_route[1] == self.original_route[-1]:
                self.route_choices = [[self.original_route[1]]]
        # TODO: change with the roadnet
        # 16x3
        # if self.id.split("_")[1] == '0' or self.id.split("_")[1] == '1':
        # porto
        if int(self.id.split("_")[1]) < 4:
            self.monitor = True
        else:
            self.monitor = False
        self.reset()

        # add route_choices
        # if self.route not in self.route_choices:
        #     self.route_choices.append(self.route)

    def exit_system(self):
        self.in_system = False

    def step(self, action, interval):
        # change to a route
        current_route = self.eng.get_vehicle_info(self.id)["route"][:-1].split(" ")
        if set(action) > set(current_route):
            self.info = self.eng.get_vehicle_info(self.id)
            return
        self.eng.set_vehicle_route(self.id, action)
        self.info = self.eng.get_vehicle_info(self.id)

    def reset(self):
        self.in_system = True


class Road(object):
    def __init__(self, road, world):
        self.id = road["id"]
        self.eng = world.eng
        self.info = road
        self.price = None
        self.last_vehicle = []
        self.history_vehicle = []
        self.lane = []

        lane_vehicles = self.eng.get_lane_vehicles()
        for i in range(len(self.info["lanes"])):
            lane = self.id + "_" + str(i)
            self.lane.append(lane)
            self.last_vehicle.extend(lane_vehicles[lane])
            self.history_vehicle.extend(lane_vehicles[lane])

        self.history_vehicle_count = len(self.history_vehicle)
        self.reset()

    def step(self, action, interval):
        # set price for a road
        self.price = (action[0] + 1) * 5
        pass

    def reset(self):
        self.price = 0
        pass
    
    def update_road_vehicle(self):
        lane_vehicles = self.eng.get_lane_vehicles()
        current_vehicles = []
        for i in range(len(self.info["lanes"])):
            lane = self.id + "_" + str(i)
            current_vehicles.extend(lane_vehicles[lane])
        self.last_vehicle = current_vehicles


class Lane(object):
    def __init__(self, id, world):
        self.id = id
        self.eng = world.eng
        self.price = None

        lane_vehicles = self.eng.get_lane_vehicles()
        self.last_vehicle = lane_vehicles[id]
        self.history_vehicle = lane_vehicles[id]
        self.length = self.get_lane_length(world)

        self.history_vehicle_count = len(self.history_vehicle)
        self.reset()

    def step(self, action, interval):
        # set price for a road
        self.price = (action[0] + 1) * 5
        pass

    def reset(self):
        self.price = 0
        pass
    
    def update_lane_vehicle(self):
        lane_vehicles = self.eng.get_lane_vehicles()
        current_vehicles = lane_vehicles[self.id]
        self.last_vehicle = current_vehicles

    def get_lane_length(self, world):
        points = world.all_roads[world.id2road[self.id[:-2]]].info["points"]
        length = pow(
            pow((points[0]["x"] - points[1]["x"]), 2)
            + pow((points[0]["y"] - points[1]["y"]), 2),
            0.5,
        )
        return length

class Route():
    def __init__(self, world, origin, destination, id):
        self.id = id
        self.idx = int(self.id.split("_")[-1])
        self.road_list = world.route["-".join([origin, destination])][self.idx]
        self.lane_list = world.get_lane_route(self.road_list)
        self.price = 0
    
    def step(self, action):
        # set price for a road
        self.price = (action[0] + 1) * 5

    def reset(self):
        self.price = 0
    

class Intersection(object):
    def __init__(self, intersection, world):
        self.id = intersection["id"]
        self.eng = world.eng

        # incoming and outgoing roads of each intersection, clock-wise order from North
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = None
        self.in_roads = None

        # links and phase information of each intersection
        self.roadlinks = []
        self.lanelinks_of_roadlink = []
        self.startlanes = []
        self.lanelinks = []
        self.phase_available_roadlinks = []
        self.phase_available_lanelinks = []
        self.phase_available_startlanes = []

        # define yellow phases, currently default to 0
        self.yellow_phase_id = [-1]
        self.yellow_phase_time = 0

        # parsing links and phases
        for roadlink in intersection["roadLinks"]:
            self.roadlinks.append((roadlink["startRoad"], roadlink["endRoad"]))
            lanelinks = []
            for lanelink in roadlink["laneLinks"]:
                startlane = (
                    roadlink["startRoad"] + "_" + str(lanelink["startLaneIndex"])
                )
                self.startlanes.append(startlane)
                endlane = roadlink["endRoad"] + "_" + str(lanelink["endLaneIndex"])
                lanelinks.append((startlane, endlane))
            self.lanelinks.extend(lanelinks)
            self.lanelinks_of_roadlink.append(lanelinks)

        self.startlanes = list(set(self.startlanes))

        phases = intersection["trafficLight"]["lightphases"]
        self.phases = [i for i in range(len(phases)) if not i in self.yellow_phase_id]
        for i in self.phases:
            phase = phases[i]
            self.phase_available_roadlinks.append(phase["availableRoadLinks"])
            phase_available_lanelinks = []
            phase_available_startlanes = []
            for roadlink_id in phase["availableRoadLinks"]:
                lanelinks_of_roadlink = self.lanelinks_of_roadlink[roadlink_id]
                phase_available_lanelinks.extend(lanelinks_of_roadlink)
                for lanelinks in lanelinks_of_roadlink:
                    phase_available_startlanes.append(lanelinks[0])
            self.phase_available_lanelinks.append(phase_available_lanelinks)
            phase_available_startlanes = list(set(phase_available_startlanes))
            self.phase_available_startlanes.append(phase_available_startlanes)

        self.reset()

    def insert_road(self, road, out):
        self.roads.append(road)
        self.outs.append(out)
        self.directions.append(_get_direction(road, out))

    def sort_roads(self, RIGHT):
        order = sorted(
            range(len(self.roads)),
            key=lambda i: (
                self.directions[i],
                self.outs[i] if RIGHT else not self.outs[i],
            ),
        )
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]

    def _change_phase(self, phase, interval):
        self.eng.set_tl_phase(self.id, phase)
        self._current_phase = phase
        self.current_phase_time = interval

    def step(self, action, interval):
        # if current phase is yellow, then continue to finish the yellow phase
        # recall self._current_phase means true phase id (including yellows)
        # self.current_phase means phase id in self.phases (excluding yellow)
        if self._current_phase in self.yellow_phase_id:
            if self.current_phase_time >= self.yellow_phase_time:
                self._change_phase(self.phases[self.action_before_yellow], interval)
                self.current_phase = self.action_before_yellow
            else:
                self.current_phase_time += interval
        else:
            if action == self.current_phase:
                self.current_phase_time += interval
            else:
                if self.yellow_phase_time > 0:
                    self._change_phase(self.yellow_phase_id[0], interval)
                    self.action_before_yellow = action
                else:
                    self._change_phase(action, interval)
                    self.current_phase = action

    def reset(self):
        # record phase info
        self.current_phase = 0  # phase id in self.phases (excluding yellow)
        # true phase id (including yellow)
        self._current_phase = self.phases[0]
        self.eng.set_tl_phase(self.id, self._current_phase)
        self.current_phase_time = 0
        self.action_before_yellow = None


class World(object):
    """
    Create a CityFlow engine and maintain informations about CityFlow world
    """

    def __init__(self, cityflow_config, thread_num, args):
        print("building world...")
        self.eng = cityflow.Engine(cityflow_config, thread_num=thread_num)
        # print(os.getcwd())
        with open(cityflow_config) as f:
            cityflow_config = json.load(f)
        self.roadnet = self._get_roadnet(cityflow_config)
        self.route = self._get_route(cityflow_config)
        # vehicles moves on the right side, currently always set to true due to CityFlow's mechanism
        self.RIGHT = True
        self.interval = cityflow_config["interval"]
        self.args = args
        self.changed_vehicle_num = {}
        self.epsilon = 1
        self.epsilon_decay = 0.99 

        # get all non virtual intersections
        self.intersections = [
            i for i in self.roadnet["intersections"] if not i["virtual"]
        ]
        self.intersection_ids = [i["id"] for i in self.intersections]
        self.all_intersections = [i for i in self.roadnet["intersections"]] 
        # create non-virtual Intersections
        print("creating intersections...")
        non_virtual_intersections = [
            i for i in self.roadnet["intersections"] if not i["virtual"]
        ]
        self.intersections = [Intersection(i, self) for i in non_virtual_intersections]
        self.intersection_ids = [i["id"] for i in non_virtual_intersections]
        self.id2intersection = {i.id: i for i in self.intersections}
        print("intersections created.")

        # id of all roads and lanes
        print("parsing roads...")
        self.all_road_ids = []
        self.all_roads = []
        self.all_lane_ids = []
        self.all_lanes = []
        self.id2lane = {}
        self.id2road = {}

        for ind, road in enumerate(self.roadnet["roads"]):

            self.id2road[road["id"]] = ind

            self.all_road_ids.append(road["id"])
            self.all_roads.append(Road(road, self))
            i = 0
            for _ in road["lanes"]:
                lane_id = road["id"] + "_" + str(i)
                self.all_lane_ids.append(lane_id)
                self.all_lanes.append(Lane(lane_id, self))
                i += 1

            iid = road["startIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, True)
            iid = road["endIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, False)
        ind = 0
        for road in self.all_roads:
            for lane in road.lane:
                self.id2lane[lane] = ind
                ind += 1
        for i in self.intersections:
            i.sort_roads(self.RIGHT)
        # print("roads parsed.")

        self.vehicles = []
        self.vehicle_ids = []
        self.id2vehicle = {}
        # TODO: change with the roadnet
        # porto
        self.vehicle_route = {"0":[0, 0, 0], "1":[0, 0, 0], "2":[0, 0, 0], "3":[0, 0, 0]}
        # others:
        # self.vehicle_route = {"0":[0, 0, 0], "1":[0, 0, 0]}

        # initializing info functions
        self.info_functions = {
            "vehicles": (lambda: self.eng.get_vehicles()),  # delete waiting==True
            "lane_count": self.eng.get_lane_vehicle_count,
            "lane_waiting_count": self.eng.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.eng.get_lane_vehicles,
            "time": self.eng.get_current_time,
            "vehicle_distance": self.eng.get_vehicle_distance,
            "pressure": self.get_pressure,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "lane_delay": self.get_lane_delay,
            "vehicle_trajectory": self.get_vehicle_trajectory,
            "history_vehicles": self.get_history_vehicles,
            "lane_vehicle_speed": self.get_lane_vehicle_speed,
            "lane_throughput": self.get_lane_throughput,
            "lane_distance": self.get_lane_distance,
            "reward_function": self.get_reward_function,
        }
        self.fns = []
        self.info = {}

        # key: vehicle_id, value: the waiting time of this vehicle since last halt.
        self.vehicle_waiting_time = {}
        # key: vehicle_id, value: [[lane_id_1, enter_time, time_spent_on_lane_1], ... , [lane_id_n, enter_time, time_spent_on_lane_n]]
        self.vehicle_trajectory = {}
        self.history_vehicles = set()

        self.lane_vehicles = self.eng.get_lane_vehicles()  # used to distance
        self.last_lane_vehicles = self.eng.get_lane_vehicles()  # used to throughput
        self.vehicle_enter_time = {key:{} for key in self.all_lane_ids}
        self.travel_times = {key:{} for key in self.all_lane_ids}

        self.train_id2road = {}
        self.train_id2allroad = {}
        self.train_id2lane = {}
        self.train_id2alllane = {}
        virtual_intersections = [i for i in self.roadnet["intersections"] if i["virtual"]]
        margin_road = []
        self.margin_lane = []
        for i in virtual_intersections:
            con_roads = i['roads']  #边缘路段ID
            margin_road.extend(con_roads)
            for r in con_roads:
                self.margin_lane.extend(self.all_roads[self.id2road[r]].lane)
        ind = 0
        index = 0
        for road_id in self.all_road_ids:
            if road_id not in margin_road:
                self.train_id2road[road_id] = ind
                self.train_id2allroad[road_id] = self.id2road[road_id]
                for lane in self.all_roads[self.id2road[road_id]].lane:
                    self.train_id2lane[lane] = index
                    index += 1
                    self.train_id2alllane[lane] = self.id2lane[lane]
                ind += 1
        self.no_train_idx, self.train_idx = self.generate_lane_train_id()
        # self.forbidden_edge, self.forbidden_train_edge = self.forbidden_edge()
        self.edge_index, self.train_edge_index = self.generate_lanelink_edge_index()
        self.large_edge_index = self.generate_large_edge_index(args.batch_size)
        print("world built.")
    
    def generate_lanelink_edge_index(self):
        edge_index = []
        train_edge_index = []
        for inter in self.intersections:
            lanelinks = inter.lanelinks
            for link in lanelinks:
                edge_index.append([self.id2lane[link[0]], self.id2lane[link[-1]]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        for inter in self.intersections:
            lanelinks = inter.lanelinks
            for link in lanelinks:
                if link[0] in self.train_id2lane and link[0] in self.train_id2lane:
                    train_edge_index.append([self.id2lane[link[0]], self.id2lane[link[-1]]])
        train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()
        return edge_index, train_edge_index

    def generate_large_edge_index(self, batch_size):
        edge_index = self.edge_index
        N = len(self.all_lane_ids)
        cat_graph = []
        for i in range(batch_size):
            cat_graph.append(edge_index)
            edge_index = edge_index + N
        large_edge_index = torch.cat(cat_graph, axis=1)
        return large_edge_index
    
    def forbidden_edge(self):
        forbiddenr2r = {}  # 禁止连续行驶的road（由cityflow不可掉头的性质得出）
        forbidden_all = []
        forbidden_train = []
        # 'id': 'road_0_1_0'
        for road in self.all_roads:
            rid = road.id
            num = road.id.split('_')
            x = int(num[1])
            y = int(num[2])
            tail = num[-1]
            if tail == "0":
                forbiddenr2r[rid] = "road_{}_{}_2".format(x + 1, y)
            elif tail == "1":
                forbiddenr2r[rid] = "road_{}_{}_3".format(x, y + 1)
            elif tail == "2":
                forbiddenr2r[rid] = "road_{}_{}_0".format(x - 1, y)
            else:
                forbiddenr2r[rid] = "road_{}_{}_1".format(x, y - 1)
        for rid in forbiddenr2r:
            if rid in self.train_id2road:
                forbidden_train.append([self.train_id2road[rid], self.train_id2road[forbiddenr2r[rid]]])
            forbidden_all.append([self.id2road[rid], self.id2road[forbiddenr2r[rid]]])
        return forbidden_all, forbidden_train

    def generate_lane_train_id(self):
        train_idx = np.array([value for key, value in self.train_id2alllane.items()])
        no_train_idx = np.array([self.id2lane[lane] for lane in self.margin_lane]) 
        return no_train_idx, train_idx

    def generate_notrain_id(self):
        idx = []
        virtual_intersections = [i for i in self.roadnet["intersections"] if i["virtual"]]
        id2road = self.id2road 
        for i in virtual_intersections:
            con_roads = i['roads']  #边缘路段ID
            for r in con_roads:
                idx.append(id2road[r])
        idx = np.array(idx)
        train_idx = np.array([value for key, value in self.train_id2allroad.items()])
        return idx, train_idx

    def generate_edge_index(self):
        '''
        return data, train_data's edges(tensor)
        '''
        edge_index = []
        train_edge_index = []
        roads = self.all_roads
        id2road = self.id2road
        non_virtual_inter = self.intersections
        all_intersections = self.all_intersections
        train_id2road = self.train_id2road
        interinfo = {}
        # train
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
        for key, value in interinfo.items():
            for road2 in value["asend"]:
                for road1 in value["asstart"]:
                    train_edge_index.append([road2, road1])
        for forbd in self.forbidden_train_edge:
            train_edge_index.remove(forbd)
        train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()
        # all
        interinfo = {} 
        for inter in all_intersections:
            interinfo[inter["id"]] = {"asstart": [], "asend": []} 
        for road in roads:
            rid = road.id
            startIntersection = road.info["startIntersection"]
            endIntersection = road.info["endIntersection"]
            road_num = id2road[rid]
            interinfo[startIntersection]["asstart"].append(road_num)
            interinfo[endIntersection]["asend"].append(road_num)
        for key, value in interinfo.items():
            for road2 in value["asend"]:
                for road1 in value["asstart"]:
                    edge_index.append([road2, road1])
        for forbd in self.forbidden_edge:
            edge_index.remove(forbd)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index, train_edge_index

    def get_reward_function(self):
        lane_reward = {}
        current_lane_vehicle_count = self.eng.get_lane_vehicle_count()
        for lane in self.all_lane_ids:
            if current_lane_vehicle_count[lane] <= 12:
                lane_reward[lane] = current_lane_vehicle_count[lane]
            else:
                lane_reward[lane] = 24 - current_lane_vehicle_count[lane]
        return lane_reward

    def get_road_vehicle_count(self):
        """
        use as state
        """
        lane_vehicle_cnt = self.eng.get_lane_vehicle_count()
        road_vehicle_count = {}
        for road in self.all_roads:
            road_vehicle_count[road.id] = 0
            for i in range(len(road.info["lanes"])):
                lane = road.id + "_" + str(i)
                road_vehicle_count[road.id] += lane_vehicle_cnt[lane]

        return road_vehicle_count
        
    def get_lane_route(self, route):
        lane_route = []
        for i in range(len(route) - 1):
            from_road = int(route[i].split("_")[-1])
            to_road = int(route[i + 1].split("_")[-1])
            if ((from_road - 1) == to_road) or (from_road == 0 and to_road == 3):
                lane_route.append(route[i] + '_2')
            elif ((from_road + 1) == to_road) or (from_road == 3 and to_road == 0):
                lane_route.append(route[i] + '_0')
            elif from_road == to_road:
                lane_route.append(route[i] + '_1')
        lane_route.append(route[-1] + '_1')
        return lane_route 

    def get_delta_toll(self):
        beta = 8
        # TODO: change with the roadnet
        # 16x3
        # maxSpeed = 16.7
        # 4x4
        # maxSpeed = 11.112
        # porto
        maxSpeed = 16.7
        free_tt = {key: self.all_lanes[self.id2lane[key]].length / maxSpeed for key in self.focus_lane}
        actual_tt = self.get_lane_average_traveltime()
        delta_toll = {key: np.array([beta * (actual_tt[key] - free_tt[key])/10]) for key in self.focus_lane}
        return delta_toll
    
    def get_lane_average_traveltime(self):
        # global_vehicles = self.eng.get_lane_vehicles()
        # vehicles = global_vehicles[self.lane]
        # current_time = self.eng.get_current_time()
        actual_tt = {key:{} for key in self.focus_lane}
        MIN_TIME = 300/16.7
        for lane in self.travel_times:
            if len(self.travel_times[lane]) != 0:
                actual_tt[lane] = np.mean([value for key, value in self.travel_times[lane].items()])
            else:
                actual_tt[lane] = MIN_TIME
        self.travel_times = {key:{} for key in self.focus_lane}
        return actual_tt
        
    def update_toll(self):
        global_vehicles = self.eng.get_lane_vehicles() #现在系统里的车
        current_time = self.get_info("time")
        for lane in self.focus_lane:
            for vehicle in global_vehicles[lane]:
                if not vehicle in self.vehicle_enter_time[lane]:  # 新进入道路的车
                    self.vehicle_enter_time[lane][vehicle] = current_time  # 新记录进时间表
            for vehicle in list(self.vehicle_enter_time[lane]):  # 所有经过道路的车（新+正+旧）
                if not vehicle in global_vehicles[lane]:  # 已经驶离的（旧）
                    self.travel_times[lane][vehicle] = current_time - self.vehicle_enter_time[lane][vehicle]
                    del self.vehicle_enter_time[lane][vehicle]  # 删去时间表相关记录

    def get_state(self):
        '''
        road-level
        vehicles_id = self.eng.get_vehicles()
        obs = {key: 0 for key in self.all_road_ids}
        for vehicle in vehicles_id:
            vec_info = self.eng.get_vehicle_info(vehicle)
            route = vec_info['route'][:-1].split(' ')
            route_len = len(route)
            for road in route:
                obs[road] += (1 / route_len)
        return obs
        '''
        obs = {key: np.zeros(10) for key in self.all_lane_ids}
        vehicles_id = self.eng.get_vehicles()
        for vehicle in vehicles_id:
            lane_route = []
            vec_info = self.eng.get_vehicle_info(vehicle)
            route = vec_info['route'][:-1].split(' ')
            for i in range(len(route) - 1):
                from_road = int(route[i].split("_")[-1])
                to_road = int(route[i + 1].split("_")[-1])
                if ((from_road - 1) == to_road) or (from_road == 0 and to_road == 3):
                    lane_route.append(route[i] + '_2')
                elif ((from_road + 1) == to_road) or (from_road == 3 and to_road == 0):
                    lane_route.append(route[i] + '_0')
                elif from_road == to_road:
                    lane_route.append(route[i] + '_1')
            lane_route.append(route[-1] + '_1') 
            for j, lane in enumerate(lane_route):
                if j < 10:
                    obs[lane][j] += 1
                else:
                    break
        return obs

    def get_road_distance(self):
        """
        返回的直接是针对路段的sum{distance}(未除以车辆数)
        >>>
        {
            road_id: distance
        }
        """
        lane_distance = self.get_lane_distance()
        road_distance = {}
        for road in self.all_roads:
            road_distance[road.id] = 0
            for i in range(len(road.info["lanes"])):
                lane = road.id + "_" + str(i)
                road_distance[road.id] += lane_distance[lane]

        return road_distance

    def get_lane_distance(self):
        detail = {}
        lane_distance = {}
        current_lane_vehicles = self.eng.get_lane_vehicles()
        old_lane_vehicles = self.lane_vehicles
        '''
        left_vehicles = {key: set(old_lane_vehicles[key]) - set(current_lane_vehicles[key]) for key in self.all_lane_ids}
        remain_vehicles = {key: set(old_lane_vehicles[key]) & set(current_lane_vehicles[key]) for key in self.all_lane_ids}
        new_vehicles = {key: set(current_lane_vehicles[key]) - set(old_lane_vehicles[key]) for key in self.all_lane_ids}
        '''
        for lane in self.all_lane_ids:
            detail[lane] = {}
            left_vehicles = set(old_lane_vehicles[lane]) - set(
                current_lane_vehicles[lane]
            )
            remain_vehicles = set(old_lane_vehicles[lane]) & set(
                current_lane_vehicles[lane]
            )
            new_vehicles = set(current_lane_vehicles[lane]) - set(
                old_lane_vehicles[lane]
            )
            lane_distance[lane] = np.array([0])
            road = lane[:-2]
            points = self.all_roads[self.id2road[road]].info["points"]
            length = pow(
                pow((points[0]["x"] - points[1]["x"]), 2)
                + pow((points[0]["y"] - points[1]["y"]), 2),
                0.5,
            )
        
            for vec_id in left_vehicles:
                vehicle = self.vehicles[self.id2vehicle[vec_id]]
                cur_dis = length - vehicle.last_distance
                lane_distance[lane] = lane_distance[lane] + cur_dis
                detail[lane][vec_id] = cur_dis
                # try:
                #     in_road = self.eng.get_vehicle_info(vehicle.id)["road"]
                # except (KeyError,RuntimeError):
                #     # vehicle.last_distance = 0
                #     continue
                # vehicle.last_distance = float(self.eng.get_vehicle_info(vehicle.id)['distance'])
            for vec_id in remain_vehicles:
                vehicle = self.vehicles[self.id2vehicle[vec_id]]
                cur_dis = (
                    float(self.eng.get_vehicle_info(vehicle.id)["distance"])
                    - vehicle.last_distance
                )
                vehicle.last_distance = float(
                    self.eng.get_vehicle_info(vehicle.id)["distance"]
                )
                lane_distance[lane] = lane_distance[lane] + cur_dis
                detail[lane][vec_id] = cur_dis
            for vec_id in new_vehicles:
                vehicle = self.vehicles[self.id2vehicle[vec_id]]
                cur_dis = float(self.eng.get_vehicle_info(vehicle.id)["distance"])
                vehicle.last_distance = float(
                    self.eng.get_vehicle_info(vehicle.id)["distance"]
                )
                lane_distance[lane] = lane_distance[lane] + cur_dis
                detail[lane][vec_id] = cur_dis

        self.lane_vehicles = current_lane_vehicles
        return lane_distance, detail

    def get_continuous_vehicle_count(self):
        continuous_vehicle_count = {}
        for road in self.all_roads:
            continuous_vehicle_count[road.id] = road.history_vehicle_count
        return continuous_vehicle_count 

    '''
    def get_lane_continuous_vehicle_count(self):
        current_lane_vehicles = self.eng.get_lane_vehicles()
        old_lane_vehicles = self.last_lane_vehicles 
        lane_vehicle_count = {}
        for lane in self.all_lane_ids:
            continuous_vehicle = list(set(current_lane_vehicles[lane]).union(set(old_lane_vehicles[lane]))) 
            lane_vehicle_count[lane] = len(continuous_vehicle)
        self.last_lane_vehicles = current_lane_vehicles
        return lane_vehicle_count
    '''

    def get_lane_throughput(self):
        current_lane_vehicles = self.eng.get_lane_vehicles()
        old_lane_vehicles = self.last_lane_vehicles
        # print("current_lane_vehicles:",current_lane_vehicles)
        # print("old_lane_vehicles:",old_lane_vehicles)
        lane_throughput = {}
        for lane in self.all_lane_ids:
            left_vehicles = set(old_lane_vehicles[lane]) - set(
                current_lane_vehicles[lane]
            )
            lane_throughput[lane] = len(left_vehicles)
            # if(lane_throughput[lane]!=0):
            #     print("lane:{}-throughput:{}".format(lane,lane_throughput[lane]))
            # print(lane,":",left_vehicles)
        self.last_lane_vehicles = current_lane_vehicles  # used to distance
        return lane_throughput

    def get_lane_vehicle_speed(self):
        # value:{lane_id:[vehicle_id]}
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_vehicle_speed = {}
        for lane in self.all_lane_ids:
            vehicle_speed = []
            if len(lane_vehicles[lane]) == 0:
                # 没有车进入该车道
                lane_vehicle_speed[lane] = 2
                continue
            for vehicle in lane_vehicles[lane]:
                vehicle_speed.append(float(self.eng.get_vehicle_info(vehicle)["speed"]))
            # print("lane:{},vehicle_speed:{}".format(lane,vehicle_speed))
            lane_vehicle_speed[lane] = np.mean(vehicle_speed) + 0.5 * len(
                [v for v in vehicle_speed if v != 0]
            )
            # print("lane:{},lane_vehicle_speed:{}".format(lane,lane_vehicle_speed[lane]))
        return lane_vehicle_speed

    def get_pressure(self):
        vehicles = self.eng.get_lane_vehicle_count()
        pressures = {}
        for i in self.intersections:
            pressure = 0
            in_lanes = []
            for road in i.in_roads:
                from_zero = (
                    (road["startIntersection"] == i.id)
                    if self.RIGHT
                    else (road["endIntersection"] == i.id)
                )
                for n in range(len(road["lanes"]))[:: (1 if from_zero else -1)]:
                    in_lanes.append(road["id"] + "_" + str(n))
            out_lanes = []
            for road in i.out_roads:
                from_zero = (
                    (road["endIntersection"] == i.id)
                    if self.RIGHT
                    else (road["startIntersection"] == i.id)
                )
                for n in range(len(road["lanes"]))[:: (1 if from_zero else -1)]:
                    out_lanes.append(road["id"] + "_" + str(n))
            for lane in vehicles.keys():
                if lane in in_lanes:
                    pressure += vehicles[lane]
                if lane in out_lanes:
                    pressure -= vehicles[lane]
            pressures[i.id] = pressure
        return pressures

    # return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
    # [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def get_vehicle_lane(self):
        # get the current lane of each vehicle. {vehicle_id: lane_id}
        vehicle_lane = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        for lane in self.all_lane_ids:
            for vehicle in lane_vehicles[lane]:
                vehicle_lane[vehicle] = lane
        return vehicle_lane

    def get_vehicle_waiting_time(self):
        # the waiting time of vehicle since last halt.
        vehicles = self.eng.get_vehicles(include_waiting=False)
        vehicle_speed = self.eng.get_vehicle_speed()
        for vehicle in vehicles:
            if vehicle not in self.vehicle_waiting_time.keys():
                self.vehicle_waiting_time[vehicle] = 0
            if vehicle_speed[vehicle] < 0.1:
                self.vehicle_waiting_time[vehicle] += 1
            else:
                self.vehicle_waiting_time[vehicle] = 0
        return self.vehicle_waiting_time

    def get_lane_waiting_time_count(self):
        # the sum of waiting times of vehicles on the lane since their last halt.
        lane_waiting_time = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        vehicle_waiting_time = self.get_vehicle_waiting_time()
        for lane in self.all_lane_ids:
            lane_waiting_time[lane] = 0
            for vehicle in lane_vehicles[lane]:
                lane_waiting_time[lane] += vehicle_waiting_time[vehicle]
        return lane_waiting_time

    def get_lane_delay(self):
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        # set speed limit to 11.11 by default
        speed_limit = 11.11
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_delay = {}
        lanes = self.all_lane_ids
        vehicle_speed = self.eng.get_vehicle_speed()

        for lane in lanes:
            vehicles = lane_vehicles[lane]
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            for vehicle in vehicles:
                speed = vehicle_speed[vehicle]
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = speed_limit
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[lane] = 1 - lane_avg_speed / speed_limit
        return lane_delay

    def get_vehicle_trajectory(self):
        # lane_id and time spent on the corresponding lane that each vehicle went through
        vehicle_lane = self.get_vehicle_lane()
        vehicles = self.eng.get_vehicles(include_waiting=False)
        for vehicle in vehicles:
            if vehicle not in self.vehicle_trajectory:
                self.vehicle_trajectory[vehicle] = [
                    [vehicle_lane[vehicle], int(self.eng.get_current_time()), 0]
                ]
            else:
                if vehicle not in vehicle_lane.keys():
                    continue
                if vehicle_lane[vehicle] == self.vehicle_trajectory[vehicle][-1][0]:
                    self.vehicle_trajectory[vehicle][-1][2] += 1
                else:
                    self.vehicle_trajectory[vehicle].append(
                        [vehicle_lane[vehicle], int(self.eng.get_current_time()), 0]
                    )
        return self.vehicle_trajectory

    def get_history_vehicles(self):
        self.history_vehicles.update(self.eng.get_vehicles())
        return self.history_vehicles

    def _get_roadnet(self, cityflow_config):
        roadnet_file = osp.join(cityflow_config["dir"], cityflow_config["roadnetFile"])
        with open(roadnet_file) as f:
            roadnet = json.load(f)
        return roadnet

    def _get_route(self, cityflow_config):
        route_file = osp.join(cityflow_config["dir"], cityflow_config["routeFile"])
        with open(route_file) as f:
            route = json.load(f)
        return route

    def subscribe(self, fns):
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if not fn in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception("info function %s not exists" % fn)

    def step(self, actions=None):
        if actions is not None:
            for i, action in enumerate(actions["tsc"]):
                self.intersections[i].step(action, self.interval)
            try:
                for j, action in enumerate(actions["cp"]):
                    self.all_lanes[j].step(action, self.interval)
            except KeyError:
                pass
            try:
                for k, action in enumerate(actions["vehicle"]):
                    if self.vehicles[k].in_system and action != []:
                        self.vehicles[k].step(action, self.interval)
            except KeyError:
                pass
        self.eng.next_step()

        '''
        lane_vehicles = self.eng.get_lane_vehicles()
        for road in self.all_roads:
            road_vehicles = []
            for i in range(len(road.info["lanes"])):
                lane = road.id + "_" + str(i)
                road_vehicles.extend(lane_vehicles[lane])
            road.history_vehicle = list(set(road.history_vehicle).union(set(road_vehicles)))
            road.history_vehicle_count = len(road.history_vehicle)
        '''

        self._update_vehicles()
        self._update_infos()

    def reset(self):
        self.eng.reset()
        self.vehicles = []
        self.vehicle_ids = []
        self.id2vehicle = {}
        self.lane_vehicles = self.eng.get_lane_vehicles()
        # TODO: change with the roadnet
        # porto
        self.vehicle_route = {"0":[0, 0, 0], "1":[0, 0, 0], "2":[0, 0, 0], "3":[0, 0, 0]}
        # others:
        # self.vehicle_route = {"0": [0, 0, 0], "1": [0, 0, 0]}

        self.vehicle_enter_time = {key:{} for key in self.all_lane_ids}
        self.travel_times = {key:{} for key in self.all_lane_ids}

        for I in self.intersections:
            I.reset()

        for V in self.vehicles:
            V.reset()

        for R in self.all_roads:
            R.reset()
        
        for L in self.all_lanes:
            L.reset()

        self._update_vehicles()
        self._update_infos()

    def _update_vehicles(self):

        # print("parsing vehicles")
        new_vehicle_ids = self.eng.get_vehicles()
        old_vehicle_ids = self.vehicle_ids
        new_entered_vehicle_ids = set(new_vehicle_ids) - set(old_vehicle_ids)
        new_left_vehicle_ids = set(old_vehicle_ids) - set(new_vehicle_ids)

        # print("new_left_vehicle_ids:",new_left_vehicle_ids)

        # update vehicle in_system
        for vec_id in new_left_vehicle_ids:
            self.vehicles[self.id2vehicle[vec_id]].exit_system()

        # add new vehicles
        for vec_id in new_entered_vehicle_ids:
            self.id2vehicle[vec_id] = len(self.vehicle_ids)
            self.vehicle_ids.append(vec_id)
            self.vehicles.append(
                Vehicle(vec_id, self.eng.get_vehicle_info(vec_id), self)
            )

        # print("vehicles parsed")

    def _update_infos(self):
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_info(self, info):
        return self.info[info]


if __name__ == "__main__":
    world = World("dataset/3x3/config.json", thread_num=1)
    # print(len(world.intersections[0].startlanes))
    print(world.intersections[0].phase_available_startlanes)
