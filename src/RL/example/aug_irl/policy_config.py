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
        self.data['restore'] = 'False'
        self.data['save'] = 'True'
        self.data['episode_num'] = 10
        self.data['total_episodes'] = 30
        self.data['epsilon'] = 0.2
        self.data['obs_dim'] = 7
        self.data['act_dim'] = 1
        self.data['lr_a'] = 1e-4
        self.data['epochs'] = 10
        self.data['batch_num'] = 20
        self.data['long_term_batch'] = 2
        self.data['reward_threshold'] = 0.001
        self.data['eval_string'] = 'dense'
        self.data['judge_string'] = 'stable'
        self.data['save_dir'] = 'train_log/model/'
        self.data['log_dir'] = 'train_log/log/'

    def get_knowledge(self):
        # knowledge
        # the list for mode choose
        self.knowledge['restore'] = ['True', 'False']
        self.knowledge['save'] = ['True', 'False']
        self.knowledge['episode_num'] = [5, 10, 20]
        self.knowledge['epsilon'] = [0.1, 0.2]
        self.knowledge['total_episodes'] = [10, 20, 30]

    def refresh(self, name=None):
        # type change
        self.data['obs_dim'] = int(self.data['obs_dim'])
        self.data['act_dim'] = int(self.data['act_dim'])

    def push(self, updata):
        # rewrite push method for some data. which is needless to show
        self.data['global_step'] = 0
        # push
        for name, value in self.data.items():
            updata[name] = value


if __name__ == '__main__':
    PolicyConfig('policy')
