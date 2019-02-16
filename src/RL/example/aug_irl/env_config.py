# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this is the test version of config:
# add path:
import sys
sys.path.append('/mnt/storage/codes/Harvey')
from RL.configure import configure


class EnvConfig(configure.SubConfig):
    def __init__(self, name):
        configure.SubConfig.__init__(self, name)
        # core config for mujoco
        self.get_knowledge()
        self.get_data()

    def get_data(self):
        self.data['source'] = 'sin'
        self.data['amplify'] = 1.0
        self.data['noise_level'] = 0.0
        self.data['run_time'] = 2
        self.data['frequency'] = 1000
        self.data['section_num'] = 10

    def get_knowledge(self):
        # knowledge
        # the list for base task choice
        self.knowledge['sin'] = ['sin']
        self.knowledge['section_num'] = [10, 20, 50, 100]

    def refresh(self, name=None):
        # attribute:
        pass

    def push(self, updata):
        # insight design:
        self.data['amplify'] = float(self.data['amplify'])
        self.data['noise_level'] = float(self.data['noise_level'])
        self.data['total_len'] = self.data['run_time'] * self.data['frequency']
        self.data['section_len'] = self.data['total_len'] / self.data['section_num']
        # push
        for name, value in self.data.items():
            updata[name] = value


if __name__ == '__main__':
    EnvConfig('environment')
