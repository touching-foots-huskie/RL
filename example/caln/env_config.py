# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this is the test version of config:
import configure
from collections import OrderedDict

class mujoco_env_config(configure.sub_config):
    def __init__(self, name):
        configure.sub_config.__init__(self, name)
        # core config for mujoco
        self.get_data()
        self.get_knowledge()

    def get_data(self):
        self.data = OrderedDict()
        self.data['attribute_num'] = 0
        self.data['base'] = ''


    def get_knowledge(self):
        # knowledge
        # the list for base task choice
        self.knowledge['base'] = ['ball', 'arm', '3darm']
        self.knowledge['attribute_0'] = ['safety', 'door', 'speed', 'force']
        # dim_information
        self.knowledge['action_dims'] = {'ball':2, 'arm':5, '3darm':4}
        self.knowledge['dim_lists'] = {'ball': '4,4', 'arm': '10,4', '3darm':'8,6', 'safety':',4',
                                       'door':',2', 'speed':',2', 'force':',2'}

    def refresh(self, name=None):
        # attribute:
        if name == 'attribute_num':
            # when update this
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


if __name__ == '__main__':
    whole_config = configure.config()
    env = configure.sub_config('environment')
    env['env_type'] = 'mujoco'
    env['env_dim'] = 4

    policy = configure.sub_config('policy')
    policy['policy_type'] = 'ppo'

    whole_config.add(env)
    whole_config.add(policy)