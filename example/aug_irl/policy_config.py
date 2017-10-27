# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# policy configures
import sys
sys.path.append('/mnt/storage/codes/Harvey')
import numpy as np
from RL.configure import configure


class PolicyConfig(configure.SubConfig):
    def __init__(self, name):
        configure.SubConfig.__init__(self, name)
        # core config for mujoco
        self.get_knowledge()
        self.get_data()

    def get_data(self):
        self.data['obs_dim'] = 7
        self.data['act_dim'] = 1
        self.data['save_dir'] = 'train_log/model/'
        self.data['log_dir'] = 'train_log/log/'

    def get_knowledge(self):
        # knowledge
        # the list for mode choose
        pass

    def refresh(self, name=None):
        # type change
        self.data['obs_dim'] = int(self.data['obs_dim'])
        self.data['act_dim'] = int(self.data['act_dim'])

    def push(self, updata):
        # rewrite push method for some data. which is needless to show
        # push
        for name, value in self.data.items():
            updata[name] = value


if __name__ == '__main__':
    PolicyConfig('policy')
