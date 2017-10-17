# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# define configuration and sub-configuration
from collections import OrderedDict
import json


class Config:
    def __init__(self):
        self.data = OrderedDict()
        self.sub_config_dict = OrderedDict()

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def add(self, sub_config):
        self.sub_config_dict[sub_config.name] = sub_config
        sub_config.upper_data = self.data
        sub_config.pull(self.data)

    def refresh(self):
        for sub_config in self.sub_config_dict.values():
            sub_config.push(self.data)

    def save(self, name):
        with open('{}.json'.format(name), 'w') as f:
            json.dump(self.data, f)


class SubConfig:
    def __init__(self, name):
        self.name = name
        self.data = OrderedDict()
        self.knowledge = OrderedDict()
        self.upper_data = OrderedDict()

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def pull(self, updata):
        # generate current data according to the updata
        # need to be overwrite
        for name in self.data.keys():
            if name in updata.keys():
                self.data[name] = updata[name]

    def push(self, updata):
        # generate more specific based on current configurations
        # save current data and generated data into updata
        # need to be overwrite
        for name, value in self.data.items():
            updata[name] = value

    def refresh(self, name=None):
        # refresh means using knowledge to update data
        pass


if __name__ == '__main__':
    whole_config = Config()
    env = SubConfig('environment')
    env['env_type'] = 'mujoco'
    env['env_dim'] = 4

    policy = SubConfig('policy')
    policy['policy_type'] = 'ppo'

    whole_config.add(env)
    whole_config.add(policy)
