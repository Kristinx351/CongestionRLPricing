class FixedHumanAgent:
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world

    def get_ob(self):
        return 0

    def get_reward(self):
        return 0

    def get_action(self, world):
        current_route = self.world.eng.get_vehicle_info(self.vehicle.id)["route"]
        return current_route[:-1].split(" ")
