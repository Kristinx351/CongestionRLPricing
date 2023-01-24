class Intersection:
    def __init__(self, iid, Point):
        self.iid = iid
        self.id_inter = "intersection_" + str(self.iid)
        self.point = Point
        self.x = Point.x
        self.y = Point.y
        self.Enterroads = []
        self.Exitroads = []
        self.roads = []  # =self.Enterroads + self.Exitroads
        self.type = len(self.Exitroads)
        self.roadLinks = []
        self.trafficLight = {}
        self.width = 15
        self.virtual = False

    def add_trafficLight(self):
        roadLinkIndices = []
        lightphases = self.process_phase()
        for i, link in enumerate(self.roadLinks):
            roadLinkIndices.append(i)
        trafficLight = {
            "roadLinkIndices": roadLinkIndices,
            "lightphases": lightphases
        }
        return trafficLight

    def process_phase(self):
        all_green = {}
        # ava_roadLinks = []
        if len(self.roadLinks) == 0:
            self.virtual = True
        # else:
            # all_green = [
            #     {
            #         "time": 20,
            #         "availableRoadLinks": [0,1]
            #     }]
        if self.type == 2:
            all_green = [
                {
                    "time": 20,
                    "availableRoadLinks": []
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        0,
                        1,
                    ]
                },
            ]

        if self.type == 3:
            all_green = [{
            "time": 20,
            "availableRoadLinks": [
              0,
              1
            ]
          },
          {
            "time": 20,
            "availableRoadLinks": [
              2,
              3
            ]
          },
          {
            "time": 20,
            "availableRoadLinks": [
              4,
              5
            ]
          },
        ]

        if self.type == 4:
            all_green = [
                {
                    "time": 5,
                    "availableRoadLinks": [
                        0,
                        3,
                        6,
                        9
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        0,
                        3,
                        6,
                        9,
                        4,
                        10
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        0,
                        3,
                        6,
                        9,
                        7,
                        1
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        0,
                        3,
                        6,
                        9,
                        2,
                        8
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        0,
                        3,
                        6,
                        9,
                        11,
                        5
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        2,
                        5,
                        8,
                        11,
                        4,
                        5
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        0,
                        3,
                        6,
                        9,
                        11,
                        10
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        0,
                        3,
                        6,
                        9,
                        7,
                        8
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        0,
                        3,
                        6,
                        9,
                        2,
                        1
                    ]
                }
            ]

        if self.type == 5:  # ALL_R; SS, SL, LL
            all_green = [
                {
                    "time": 20,
                    "availableRoadLinks": [
                        0,
                        1,
                        2,
                        3
                    ]
                },

                {
                    "time": 20,
                    "availableRoadLinks": [
                        4,
                        5,
                        6,
                        7
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        8,
                        9,
                        10,
                        11
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        12,
                        13,
                        14,
                        15
                    ]
                },
                {
                    "time": 20,
                    "availableRoadLinks": [
                        16,
                        17,
                        18,
                        19
                    ]
                }
            ]
        #
        #     # all_red = {
        #     #     "time": 20,
        #     #     "availableRoadLinks": []
        #     # }

        # all_green = [
        #     {
        #         "time": 5,
        #         "availableRoadLinks": [
        #             0
        #         ]
        #     }]
        return all_green

        # def is_inter_virtual(self):
        #     return False

        # def get_trafficLight(self):
        #     lightphases = self.get_lightphases()
        #     trafficLight = {
        #         "roadLinkIndices":[],
        #         "lightphases": lightphases
        #     }
        #     return trafficLight
