import numpy as np


class Price_Agent():
    def __init__(self, lane, world, R, beta):
        self.toll = 5
        self.lane = lane.id
        self.world = world
        self.vehicle_enter_time = {}
        self.travel_times = {}
        self.R = R
        self.beta = beta
        self.road_length = lane.length
    '''
    def get_ob(self):
        return self.ob_generator.generate(self.ob_generator.average)

    def get_reward(self):
        reward = self.reward_generator.generate(self.reward_generator.average)
        assert len(reward) == 1
        return reward
    '''

    def get_action(self, i, id):
        if i == 0:
            return np.array([5])

        else:
            R = self.R # 10e-4  # smooth
            beta = self.beta #4  # proportional coefficient
            # TODO: change with the roadnet
            # 16x3
            maxSpeed = 16.7
            # 4x4
            maxSpeed = 11.12
            # porto
            maxSpeed = 16.7
            free_tt = self.road_length / maxSpeed
            actual_tt = self.get_lane_average_traveltime(id)
            delta_toll = (1-R) * self.toll + R * beta * (actual_tt - free_tt)
            # delta_toll = beta * (actual_tt - free_tt)/10
           #  print("delta = ", actual_tt-free_tt, "\tdelta_toll = ", delta_toll, '\n')
            self.toll = delta_toll
            return np.array([delta_toll])

    def update(self):
        global_vehicles = self.world.eng.get_lane_vehicles()
        vehicles = global_vehicles[self.lane]
        # print(vehicles)
        # print(vehicles) # list
        current_time = self.world.get_info("time")

        for vehicle in vehicles:
            if not vehicle in self.vehicle_enter_time:  # 新进入道路的车
                self.vehicle_enter_time[vehicle] = current_time  # 新记录进时间表

        for vehicle in list(self.vehicle_enter_time):  # 所有经过道路的车（新+正+旧）
            if not vehicle in vehicles:  # 已经驶离的（旧）
                # print(vehicle,":",current_time,"-",self.vehicle_enter_time[vehicle])
                self.travel_times[vehicle] = current_time - self.vehicle_enter_time[vehicle]
                del self.vehicle_enter_time[vehicle]  # 删去时间表相关记录
        # print(self.vehicle_enter_time) # dictionary

    def get_lane_average_traveltime(self, id):
        global_vehicles = self.world.eng.get_lane_vehicles()
        vehicles = global_vehicles[self.lane]
        # print(vehicles)
        # print(vehicles) # list
        current_time = self.world.get_info("time")

        for vehicle in vehicles:
            if not vehicle in self.vehicle_enter_time:  # 新进入道路的车
                self.vehicle_enter_time[vehicle] = current_time  # 新记录进时间表

        for vehicle in list(self.vehicle_enter_time):  # 所有经过道路的车（新+正+旧）
            if not vehicle in vehicles:  # 已经驶离的（旧）
                # print(vehicle,":",current_time,"-",self.vehicle_enter_time[vehicle])
                self.travel_times[vehicle] = current_time - self.vehicle_enter_time[vehicle]
                del self.vehicle_enter_time[vehicle]  # 删去时间表相关记录
        # print(self.vehicle_enter_time) # dictionary

        if len(self.travel_times) != 0:
            # print(len(self.travel_times), "in record;")
            actual_tt = sum(value for key, value in self.travel_times.items()) / len(self.travel_times)
        else:
            # print("no vehicles!")
            actual_tt = 300/16.7
        # print("eveluated tt of lane", id, ":\t", actual_tt)
        # self.vehicle_enter_time = {}
        self.travel_times = {}
        return actual_tt




