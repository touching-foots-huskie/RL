# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this is the test version of config:
# add path:
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import RL as rl
from RL.configure import configure
# utils
from collections import OrderedDict

class env_config(configure.sub_config):
    def __init__(self, name):
        configure.sub_config.__init__(self, name)
        # core config for mujoco
        self.get_knowledge()
        self.get_data()

    def get_data(self):
        self.data['environment'] = ''
        self.data['attribute_num'] = 0
        self.data['base'] = ''

    def get_knowledge(self):
        # knowledge
        # the list for base task choice
        self.knowledge['base'] = ['ball', 'arm', '3darm']
        self.knowledge['attribute_0'] = ['safety', 'door', 'speed', 'force']
        # dim_information
        self.knowledge['action_dims'] = {'ball': 2, 'arm':5, '3darm':5}
        self.knowledge['dim_lists'] = {'ball': '4,4', 'arm': '10,4', '3darm': '10,6', 'safety': ',4',
                                       'door': ',2', 'speed': ',2', 'force': ',2'}

    def refresh(self, name=None):
        # attribute:
        if name == 'attribute_num':
            try:
                self.data['attribute_num'] = int(self.data['attribute_num'])
                for i in range(self.data['attribute_num']):
                    self.data['attribute_{}'.format(i)] = ''
                    # add dynamic knowledge
                    self.knowledge['attribute_{}'.format(i)] = self.knowledge['attribute_0']

                num = self.data['attribute_num']

                while ('attribute_{}'.format(num) in self.data):
                    del self.data['attribute_{}'.format(num)]
                    num += 1
            except:
                print('attribute_num is not int')

        elif name == 'base':
            # when chosen base:
            self.data['action_dim'] = self.knowledge['action_dims'][self.data['base']]
            self.data['dim_list'] = self.knowledge['dim_lists'][self.data['base']]

        elif name[:9] == 'attribute':
            # then dim
            i = 0
            try:
                self.data['dim_list'] = self.knowledge['dim_lists'][self.data['base']]
                while('attribute_{}'.format(i) in self.data.keys()):
                    self.data['dim_list'] += self.knowledge['dim_lists'][self.data['attribute_{}'.format(i)]]
                    i += 1
            except:
                print('Choose base first')
        else:
            pass

        name_list = self.data['base']
        i = 0
        while ('attribute_{}'.format(i) in self.data.keys()):
            name_list += ',{}'.format(self.data['attribute_{}'.format(i)])
            i += 1
        self.data['environment'] = name_list

    def push(self, updata):
        self.data['reset_from_pool'] = False
        # push
        for name, value in self.data.items():
            updata[name] = value


if __name__ == '__main__':
    env_config('environment')
