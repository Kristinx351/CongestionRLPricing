import torch
import torch.nn.functional as F
import numpy as np
class HumanEcoAgent:
    """
    Agent using Economic model for user
    """

    def __init__(self, vehicle, world):

        self.vehicle = vehicle
        self.world = world

    def get_ob(self):
        lane_vehicle_count = self.world.eng.get_lane_vehicle_count()
        # vehicle_road=self.world.eng.get_vehicle_info(self.vehicle.id).road
        road_vehicle_count = {}
        # self.world.roadnet["roads"]是所有路段(dict)
        lane_count = len(self.world.roadnet["roads"][0]["lanes"])
        for road in self.world.roadnet["roads"]:
            rvc = 0
            for i in range(lane_count):
                lane_id = road["id"] + "_" + str(i)
                rvc = rvc + lane_vehicle_count[lane_id]
            road_vehicle_count[road["id"]] = rvc
        return road_vehicle_count

    def get_action(self, world):
        # todo - get vehicle route time
        # lane_waiting_count = self.world.get_info("lane_waiting_count")
        # road_count = self.get_ob()
        # route_choices = self.vehicle.route_choices
        current_route = self.world.eng.get_vehicle_info(self.vehicle.id)["route"][:-1].split(" ")
        if len(current_route) <= 2:
            return current_route
        else:
            route_choices = world.route["-".join([current_route[1], current_route[-1]])]
        # if self.vehicle.monitor == False:
        #     return current_route[:-1].split(' ')
        '''
        try:
            # id of current road
            current_link = self.world.eng.get_vehicle_info(self.vehicle.id)["road"]
        except KeyError:
            return current_route[:-1].split(" ")
        # check whether it is at the origin link
        if current_link != self.vehicle.original_route[0]:
            return current_route[:-1].split(" ")
        '''
        # calculate three route cost
        r = []
        cost = []
        for i in range(len(route_choices)):
            # 某条route r[i]
            r.append(route_choices[i][0:])
            cost.append(self.cal_cost(world.get_lane_route(r[i])))

        # select one route
        cost = - torch.FloatTensor(cost)
        pro = F.softmax(cost / 50, dim=0)
        choice = np.random.choice(3,1,p=np.array(pro))[0]
        # mincost = min(cost)
        # print("{} choose the route{}".format(self.vehicle.id, cost.index(mincost)))
        # self.vehicle.route = r[cost.index(mincost)]  # 更新所选路线
        self.vehicle.route = r[choice]
        world.vehicle_route[self.vehicle.id.split("_")[1]][choice] += 1
        return r[choice]

    def cal_cost(self, r):

        time_coef = 1
        a = 0.2
        b = 0
        c = 1

        total_fuel = 0
        total_time = 0
        total_cost = 0
        for link_id in r:
            # link_cnt = lc[link_id]
            points = self.world.all_roads[self.world.id2road[link_id[:-2]]].info["points"]
            length = pow(
                pow((points[0]["x"] - points[1]["x"]), 2)
                + pow((points[0]["y"] - points[1]["y"]), 2),
                0.5,
            )
            total_fuel += length / 1000 * 2.4
            # total_time += link_cnt * time_coef
            # 路程开销
            total_cost += self.world.all_lanes[self.world.id2lane[link_id]].price

        # 输出查看
        # if(total_cost!=0):
        #     print("vehicle:{} route is {},total_fuel is {},total_time is {},total_cost is {}".format(self.vehicle.id,r,total_fuel,total_time,total_cost))

        return a * total_fuel + c * total_cost

    def get_reward(self):
        return None
