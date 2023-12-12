import json
from os import name


def generate_mask(roadnet_config, path_to_route_output):
    """
    对每个road都有一个mask，内容为对于当前road而言的node_feature，每个边的权重
    初始设置为与其相连的边mask为0.5，其余为0
    >>>mask:
    {
        road_id:{
            road_id: weight
        }
    }
    """
    mask = {}
    path_to_roadnet_input = roadnet_config
    path_to_route_output = path_to_route_output
    roadnet_json = json.load(open(path_to_roadnet_input, "r"))
    roads = roadnet_json["roads"]
    intersections = roadnet_json["intersections"]

    inter2road = {}  # 与inter相连的road
    startIntersection = {}  # 每个road开始的inter
    endIntersection = {}  # 每个road结束的inter
    for road in roads:
        endIntersection[road["id"]] = road["endIntersection"]
        startIntersection[road["id"]] = road["startIntersection"]

    for inter in intersections:
        inter2road[inter["id"]] = inter["roads"]

    for road in roads:
        id = road["id"]
        mask[id] = {}
        link_road = []
        # 寻找相邻路段
        link_road.extend(inter2road[endIntersection[id]])
        link_road.extend(inter2road[startIntersection[id]])
        link_road = list(set(link_road))
        for road in roads:
            subid = road["id"]
            if subid == id:
                mask[id][subid] = 1
            elif subid in link_road:
                mask[id][subid] = 0.5
            else:
                mask[id][subid] = 0

    # 导出
    json.dump(mask, open(path_to_route_output, "w"), indent=2)


if __name__ == "__main__":
    generate_mask("dataset/3x3/roadnet.json", "dataset/3x3/mask.json")
