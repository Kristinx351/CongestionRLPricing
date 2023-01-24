import numpy as np
import networkx as nx
import pandas as pd
import math
from sympy import Point, Line, Segment, Circle, intersection
from Roadnet_component.Intersection import Intersection
from Roadnet_component.Lane import Lane


class Road:
    def __init__(self, start_inter, end_inter):
        self.startIntersection = start_inter
        self.endIntersection = end_inter
        self.start_point = self.startIntersection.point
        self.end_point = self.endIntersection.point
        self.iid = self.startIntersection.iid
        self.id_road = "road_" + str(self.iid) + "_" + str(len(start_inter.Enterroads))
        self.length = self.start_point.distance(self.end_point)
        self.direction = self.get_direction()
        self.line = Line(self.start_point, self.end_point)
        # self.vector = Segment(self.start_point, self.end_point)
        self.in_laneLink_point, self.out_laneLink_point = self.get_inout_laneLink_point()
        self.lane_num = 3
        self.lanes = self.add_lanes()
        self.dic_lanes = self.get_roadlanes()
        self.angle = self.get_clockwise_angle(self.start_point, self.end_point)

    def get_clockwise_angle(self, p2, p1):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x = x2 - x1
        y = y2 - y1
        x0 = 1
        y0 = 0
        dot = x * x0 + y * y0
        det = x * y0 - y * x0
        theta = math.atan2(det, dot)
        theta = theta if theta > 0 else 2 * np.pi + theta
        return theta

    def add_lanes(self):
        lanes = []
        for i in range(0, 3):
            lanes.append(Lane(self.in_laneLink_point[i], self.out_laneLink_point[i], i))
        return lanes

    def get_lane_basePoint(self, inter_P1, inter_P2):
        # Todo get laneLlink_base_Point(middle of a road)
        # inter_P1 = self.start_point
        # inter_P2 = self.end_point
        p1, p2 = inter_P1, inter_P2
        inter_line = Line(p1, p2)
        inter_vector = Line(p1, p2)
        c = Circle(p1, 15)
        p_1, p_2 = intersection(inter_line, c)
        p11 = Point(p_1.x, p_1.y, evaluate=False)  # X Y should not be fraction
        p22 = Point(p_2.x, p_2.y, evaluate=False)
        l1 = Line(p1, p11)
        l2 = Line(p1, p22)
        # print("p11:", p11.x, p11.y)
        # print("p22:", p22.x, p22.y)

        angle_1 = inter_vector.angle_between(l1)
        angle_2 = inter_vector.angle_between(l2)
        # print("angle_1:", angle_1)
        # print("angle_2:", angle_2)
        # TODO: Imaginary numbers appear(e-8*I)
        if abs(angle_1) < math.pi / 2:
            return p11
        else:
            return p22

    def get_inout_laneLink_point(self):
        line_road = self.line
        in_laneLink_point = []
        out_laneLink_point = []
        in_basePoint = self.get_lane_basePoint(self.start_point, self.end_point)
        out_basePoint = self.get_lane_basePoint(self.end_point, self.start_point)

        for R in list([2, 6, 10]):  # lane width: 4
            in_laneLink_point.append(self.get_in_laneLink_point(line_road, in_basePoint, R))
            out_laneLink_point.append(self.get_out_laneLink_point(line_road, out_basePoint, R))

        return in_laneLink_point, out_laneLink_point

    def get_in_laneLink_point(self, line_road, basePoint, R):
        c = Circle(basePoint, R)  # lane_width = 4
        p_1, p_2 = intersection(line_road, c)
        p1 = Point(p_1.x, p_1.y, evaluate=False)
        p2 = Point(p_2.x, p_2.y, evaluate=False)
        if (self.is_laneindex_in(basePoint, p2)):
            return p2
        else:
            return p1

    def get_out_laneLink_point(self, line_road, basePoint, R):
        c = Circle(basePoint, R)  # lane_width = 4
        p_1, p_2 = intersection(line_road, c)
        p1 = Point(p_1.x, p_1.y, evaluate=False)
        p2 = Point(p_2.x, p_2.y, evaluate=False)

        if (self.is_laneindex_in(basePoint, p2)):
            return p1
        else:
            return p2

    def is_laneindex_in(self, inter_p, p):
        # e = Segment(Point(0,0), Point(1,0))
        angle_0 = math.atan2(inter_p.y, inter_p.x) * 180 / math.pi
        angle_0 = angle_0 if angle_0 > 0 else 2 * math.pi + angle_0
        angle = math.atan2(p.y, p.x) * 180 / math.pi - angle_0
        angle = angle if angle > 0 else 2 * math.pi + angle

        if angle_0 >= 0 and angle_0 < math.pi / 2:
            if angle < -math.pi:
                return False  # out_lane
        if angle_0 >= math.pi / 2 and angle_0 < 3 * math.pi / 2:
            if angle > 0:
                return False  # out_lane
        else:
            if angle < math.pi:
                return False

    def classify_laneLink(self, road):
        # Todo get laneindex_direction : "in"--get into the intersection, TRUE
        pass

    def get_direction(self):
        start_point = self.start_point
        end_point = self.end_point
        angle = self.get_clockwise_angle(start_point, end_point)
        # print("angle:", angle)
        if (angle >= 7 * math.pi / 4 and angle < math.pi / 4):
            return 0  # right
        elif (angle >= math.pi / 4 and angle < math.pi / 4 * 3):
            return 1  # up
        elif (angle < 7 * math.pi / 4 and angle >= 5 * math.pi / 4 ):
            return 3  # down
        else:
            return 2  # left

    def get_ab_angle(self, pre_point, now_point):  # tell the direction from inter A to B (edge:(A,B)
        pre_angle = math.atan2(pre_point.y, pre_point.x) * 180 / math.pi
        now_angle = math.atan2(now_point.y, now_point.x) * 180 / math.pi
        angle_diff = now_angle - pre_angle
        if (angle_diff < 0):  # make sure the angle >0
            angle_diff += math.pi * 2
        return angle_diff

    def get_roadlanes(self):
        lanes = []
        for i in range(self.lane_num):
            lanes.append(
                {
                    "width": 4,
                    "maxSpeed": 16.67
                }
            )
        return lanes
