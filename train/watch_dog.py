# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# watch_dog is used to ensure the training process doesn't


class WatchDog:
    def __init__(self, max_steps=100, min_change=0.1):
        self.max_steps = max_steps
        self.min_change = min_change
        self.global_step = 0
        self.history_record = 0
        self.history_performance = 0

    def go(self):
        self.global_step += 1

    def refresh(self, performance=0):
        self.history_record = self.global_step
        self.history_performance = performance

    def check(self, performance):
        # time go
        self.go()
        clock = self.global_step - self.history_record
        if clock > self.max_steps:
            if (performance - self.history_performance) > self.min_change:
                # success:
                self.refresh(performance)
                return True
            else:
                # fail : refresh with 0
                self.refresh()
                return False

        else:
            return True
