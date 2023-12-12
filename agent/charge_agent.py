
class Charge_Agent():
    def __init__(self, action_space):
        self.action_space = action_space
    '''
    def get_ob(self):
        return self.ob_generator.generate(self.ob_generator.average)

    def get_reward(self):
        reward = self.reward_generator.generate(self.reward_generator.average)
        assert len(reward) == 1
        return reward
    '''
    def get_action(self):
        return self.action_space.sample()

