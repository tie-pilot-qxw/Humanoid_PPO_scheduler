class ConfigScheduler:
    def __init__(self, class_name, config, iteration, gap, operation = None):
        self.flag = False
        if operation is not None:
            self.stop = True
        else:
            self.stop = False
        self.config = config
        self.gap = gap
        self.steps = 0
        self.type = class_name
        self.items = [attr for attr in dir(self.type) if not callable(getattr(self.type, attr)) and not attr.startswith("__")]
        if type(iteration) == self.type:
            self.iteration = iteration
        elif type(iteration) == int:
            self.iteration = self.type()
            for attr in self.items:
                setattr(self.iteration, attr, iteration)
        else:
            raise ValueError('iteration type is not correct')

    def step(self):
        # update reward
        if self.flag:
            return False
        self.steps += 1
        flag = False
        for attr in self.items:
            iter = self.iteration.__getattribute__(attr)
            if iter <= 0:
                continue
            if self.steps % int(iter) == 0:
                flag = True
                if self.stop:
                    self.flag = True
                old_reward = self.config.__getattribute__(attr)
                add = self.gap.__getattribute__(attr)
                if type(old_reward) == list:
                    setattr(self.config, attr, [old_reward[i] + add[i] for i in range(len(old_reward))])
                else:
                    setattr(self.config, attr, old_reward + add)
        return flag

class scales(object):
    tracking_ang_vel = None
    tracking_lin_vel = None
    def __init__(self, tracking_lin_vel = None, tracking_ang_vel = None):
        self.tracking_lin_vel = tracking_lin_vel
        self.tracking_ang_vel = tracking_ang_vel

def main():
    reward = scales(1,2)
    gap = scales(2,1)
    iteration = scales(6,3)

    scheduler = ConfigScheduler(type(reward), reward, iteration, gap)
    print(reward.tracking_lin_vel, reward.tracking_ang_vel)
    for i in range(10):
        scheduler.step()
        print(reward.tracking_lin_vel, reward.tracking_ang_vel)

if __name__ == '__main__':
    main()