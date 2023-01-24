from math import *
import argparse
import json
from Roadnet_component.Net import Net

parser = argparse.ArgumentParser(description="data generator")
parser.add_argument(
    "--roadnetfile",
    type=str,
    default="dataset/porto_roadnet/1_0.json",
    help="cityflow format roadnet file",
)
parser.add_argument(
    "--flowfile",
    type=str,
    default="flow_data/porto_roadnet/flow.json",
    help="cityflow format flow file",
)
parser.add_argument(
    "--routesfile",
    type=str,
    default="dataset/porto_roadnet/optional_routes.json",
    help="routes file",
)

roadLink_type = ["turn_left", "go_straight", "turn_right"]

interval = 1.0

# The direction of intersections: {Intersection_ID: (longitude, latitude)}
node2coord = {0: (41.156085, -8.621596), 1: (41.156553, -8.620813), 2: (41.156319, -8.619365),
              3: (41.154768, -8.613357), 4: (41.154833, -8.609709), 5: (41.155859, -8.604613),
              6: (41.156069, -8.602478), 7: (41.155561, -8.620835), 8: (41.155157, -8.619054),
              9: (41.153694, -8.613250), 10: (41.153566, -8.610020), 11: (41.153534, -8.605063),
              12: (41.154835, -8.604687), 13: (41.154617, -8.603078), 14: (41.154585, -8.602499),
              15: (41.153260, -8.602435), 16: (41.153317, -8.600675), 17: (41.150264, -8.623071),
              18: (41.150840, -8.621002), 19: (41.151543, -8.617429), 20: (41.149919, -8.614854),
              21: (41.150541, -8.610595), 22: (41.150113, -8.607183), 23: (41.150975, -8.606685),
              24: (41.150951, -8.604297), 25: (41.151032, -8.602241), 26: (41.151099, -8.600099),
              27: (41.149442, -8.604117), 28: (41.149015, -8.602073), 29: (41.148474, -8.600217),
              30: (41.147729, -8.623233), 31: (41.147623, -8.620364), 32: (41.148075, -8.617982),
              33: (41.148287, -8.615142), 34: (41.147950, -8.611267), 35: (41.149271, -8.610788),
              36: (41.148794, -8.607816), 37: (41.147658, -8.603953), 38: (41.147117, -8.601881),
              39: (41.147448, -8.608640), 40: (41.147018, -8.606773), 41: (41.146447, -8.604040),
              42: (41.145700, -8.622641), 43: (41.146055, -8.620152), 44: (41.146103, -8.617727),
              45: (41.145771, -8.614895), 46: (41.146038, -8.611140), 47: (41.146135, -8.606999),
              48: (41.145683, -8.605218), 49: (41.145222, -8.604317), 50: (41.145521, -8.603180),
              51: (41.145715, -8.601721), 52: (41.145982, -8.598223), 53: (41.143812, -8.617665),
              54: (41.142630, -8.613983), 55: (41.144965, -8.610893), 56: (41.143252, -8.610260),
              57: (41.142961, -8.608125), 58: (41.144795, -8.607052), 59: (41.142427, -8.606290),
              60: (41.144059, -8.604906), 61: (41.143065, -8.601977), 62: (41.143228, -8.621179),
              63: (41.141375, -8.614109), 64: (41.140820, -8.609724), 65: (41.142016, -8.601752),
              66: (41.141224, -8.598114), 67: (41.139932, -8.609429), 68: (41.141081, -8.601611),
              69: (41.140721, -8.616597)}
# the list of edges to represent the connections of intersections
edges = [(0, 7),
         (1, 7),
         (7, 8),
         (2, 8),
         (8, 9),
         (3, 9),
         (9, 10),
         (4, 10),
         (10, 11),
         (12, 11),
         (5, 12),
         (13, 12),
         (6, 13),
         (13, 14),
         (6, 14),
         (7, 18),
         (17, 18),
         (18, 19),
         (8, 19),
         (9, 20),
         (21, 20),
         (10, 21),
         (21, 22),
         (11, 23),
         (22, 23),
         (23, 24),
         (13, 24),
         (24, 25),
         (14, 15),
         (15, 25),
         (16, 15),
         (26, 25),
         (22, 27),
         (24, 27),
         (27, 28),
         (25, 28),
         (28, 29),
         (18, 31),
         (30, 31),
         (31, 32),
         (32, 33),
         (20, 33),
         (33, 34),
         (21, 35),
         (34, 35),
         (35, 36),
         (22, 36),
         (36, 37),
         (27, 37),
         (37, 38),
         (28, 38),
         (29, 38),
         (34, 39),
         (36, 39),
         (39, 40),
         (40, 41),
         (41, 38),
         (37, 41),
         (31, 43),
         (42, 43),
         (43, 44),
         (32, 44),
         (53, 44),
         (33, 45),
         (53, 45),
         (34, 46),
         (45, 46),
         (39, 46),
         (47, 46),
         (40, 47),
         (41, 48),
         (46, 55),
         (47, 58),
         (48, 47),
         (48, 49),
         (50, 49),
         (50, 51),
         (38, 51),
         (51, 52),
         (53, 54),
         (55, 54),
         (55, 56),
         (56, 57),
         (58, 57),
         (57, 59),
         (58, 60),
         (59, 60),
         (49, 60),
         (60, 61),
         (50, 61),
         (62, 69),
         (69, 63),
         (54, 63),
         (63, 64),
         (67, 64),
         (64, 65),
         (61, 65),
         (66, 65),
         (68, 65),
         (9, 19)]

print("---------- there is %d roads in the net---------" % (len(edges) * 2))

# nx.all_shortest_paths(G, source=0, target=2)


def get_final_intersections(net):
    # edges = net.getEdges()
    intersections = net.intersections
    final_intersections = []
    for inter in intersections:
        roads = []
        for r in inter.roads:
            roads.append(r.id_road)
        # roadlinks = inter.roadLinks
        trafficLight = inter.trafficLight
        intersection = {
            "id": inter.id_inter,
            "point": {
                "x": inter.x,
                "y": inter.y
            },
            "width": inter.width,
            "roads": roads,
            "roadLinks": inter.roadLinks,
            "trafficLight": trafficLight,
            "virtual": inter.virtual  # dead_end is "virual"
        }
        if intersection["roads"] != []:
            final_intersections.append(intersection)
            # final_intersections.append("\n")
        else:
            print("%s doesnt have any roads!" % (intersection["id"]))
    return final_intersections


def get_final_roads(net):
    roads = net.roads
    # print(len(net.roads))
    final_roads = []
    for road in roads:
        start_intersection = road.startIntersection
        end_intersection = road.endIntersection
        road = {
            # id of road
            "id": road.id_road,
            # points along the road which describe the shape of the road
            "points": [
                {
                    "x": float(start_intersection.x),
                    "y": float(start_intersection.y)
                },
                {
                    "x": float(end_intersection.x),
                    "y": float(end_intersection.y)
                }
            ],
            # property of each lane
            "lanes": road.dic_lanes,
            # id of start intersection
            "startIntersection": start_intersection.id_inter,
            "endIntersection": end_intersection.id_inter,
        }
        final_roads.append(road)
        # print(len(road))
        # final_roads.append("\n")
    return final_roads


def get_final_flows(net):
    final_flows = []
    for flow in net.flows:
        if len(flow.route)<=1:
            continue
        else:
            # Set flow parameters here
            flow = {
                "vehicle": {
                    "length": 5.0,
                    "width": 2.0,
                    "maxPosAcc": 2.0,
                    "maxNegAcc": 4.5,
                    "usualPosAcc": 2.0,
                    "usualNegAcc": 4.5,
                    "minGap": 2.5,
                    "maxSpeed": 11.111,
                    "headwayTime": 2
                },
                "route": flow.route,
                "interval": interval,
                "startTime": flow.startTime,
                "endTime": flow.endTime
            }
            final_flows.append(flow)

    return final_flows


def get_final_routes(net):
    final_routes = {}
    for flow in net.flows:
        route_1, route_2, route_3 = net.get_optional_routes(flow)
        # TODO: filter the route which only contains one road
        print("route_1:", route_1)
        if len(route_1) <= 1:
            continue
        print("route_1:", route_1[1:])
        route_key = "{}-{}".format(route_1[1], route_1[-1])
        final_routes[route_key] = [route_1[1:]]
        if route_2 == []:
            final_routes[route_key].append(route_1[1:])
            final_routes[route_key].append(route_1[1:])
        else:
            final_routes[route_key].append(route_2[1:])
            if route_3 == []:
                final_routes[route_key].append(route_2[1:])
            else:
                final_routes[route_key].append(route_3[1:])

    return final_routes


def main(args):
    net = Net(node2coord, edges)
    final_intersections = get_final_intersections(net)
    for intersection in final_intersections:
        if intersection["virtual"]:
            intersection["roadLinks"] = []
    final_roads = get_final_roads(net)
    final_flows = get_final_flows(net)

    roadnet_result = {
        "intersections": final_intersections,
        "roads": final_roads
    }
    flow_result = final_flows

    f1 = open(args.roadnetfile, "w")
    json.dump(str(roadnet_result), f1, indent=4)
    f1.close()
    print("Cityflow net file generated successfully!")

    f2 = open(args.flowfile, "w")
    json.dump(str(flow_result), f2, indent=4)
    f2.close()
    print("Cityflow flow file generated successfully!")

    routes_result = {}
    routes_result = get_final_routes(net)

    f3 = open(args.routesfile, "w")
    json.dump(str(routes_result), f3, indent=4)
    f3.close()
    print("Cityflow routes file generated successfully!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
