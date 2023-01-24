class Lane:
    def __init__(self, p1, p2, i):
        self.index = i  # "0": left,"1": straight, "2":right
        self.start_point = p1
        self.end_point = p2


    # def get_index(self):
    #     d = self.point.distance(self.basePoint)
    #     if d < 4:
    #         index = 0
    #     elif d > 8:
    #         index = 2
    #     else:
    #         index = 1
    #     return index
    #
