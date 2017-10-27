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

    def get_knowledge(self):
        # knowledge
        # the list for base task choice
        self.knowledge['sin'] = ['sin']

    def refresh(self, name=None):
        # attribute:
        self.data['amplify'] = float(self.data['amplify'])
        self.data['noise_level'] = float(self.data['noise_level'])

    def push(self, updata):
        # push
        for name, value in self.data.items():
            updata[name] = value


if __name__ == '__main__':
    EnvConfig('environment')
