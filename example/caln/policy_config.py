# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# policy configures
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import RL as rl
import numpy as np
from RL.configure import configure
from collections import OrderedDict


class policy_config(configure.sub_config):
    def __init__(self, name):
        configure.sub_config.__init__(self, name)
        # core config for mujoco
        self.get_knowledge()
        self.get_data()

    def get_data(self):
        self.data['mode'] = ''
        self.data['system_mark'] = ''
        self.data['eval_string'] = 'sum'
        self.data['update_string'] = 'curriculum'
        self.data['judge_string'] = 'curriculum'
        self.data['save_dir'] = 'train_log/model/'
        self.data['log_dir'] = 'train_log/log/'
        self.data['training_mark'] = ''
        self.data['epsilon'] = 0.05
        self.data['epochs'] = 20
        self.data['max_iter_num'] = 10
        self.data['batch_num'] = 256
        self.data['batch_epochs'] = 40
        self.data['long_term_batch'] = 10  # 10's average more than 10 can step into another level
        self.data['total_episodes'] = 1000
        self.data['random_level'] = 0.1
        self.data['filter'] = 'True'
        self.data['filter_threshold'] = 0.5
        self.data['filter_ratio'] = 0.5
        self.data['lambda'] = 1.1
        self.data['gamma'] = 0.80
        self.data['lam'] = 0.9
        self.data['reward_threshold'] = 0.9

    def get_knowledge(self):
        # knowledge
        # the list for mode choose
        self.knowledge['mode'] = ['task_train', 'attribute_train', 'weight_train',
                                  'test']
        self.knowledge['eval_string'] = ['sum', 'dis_sum']
        self.knowledge['update_string'] = ['curriculum', 'reverse_curriculum', 'stable']
        self.knowledge['judge_string'] = ['curriculum', 'reverse_curriculum', 'stable']
        # param choice:
        self.knowledge['epsilon'] = [0.2, 0.1, 0.05]
        self.knowledge['epochs'] = [20, 10, 5, 40]
        self.knowledge['max_iter_num'] = [10, 20, 40]
        self.knowledge['batch_num'] = [256, 128, 64, 32]
        self.knowledge['batch_epochs'] = [40, 80, 120, 20]
        self.knowledge['long_term_batch'] = [10, 20]
        self.knowledge['total_episodes'] = [1000, 2000, 5000, 10000]
        self.knowledge['random_level'] = [0.1, 1.0, 10.0, 0.0]
        # training method:
        self.knowledge['filter'] = ['True', 'False']
        # learning rate:
        self.knowledge['lr_cs'] = {'ball': 1e-2/np.sqrt(np.sqrt(200)),
                                   'arm': 1e-2/np.sqrt(np.sqrt(500)), '3darm': 1e-2/np.sqrt(np.sqrt(500))}
        self.knowledge['lr_ts'] = {'ball': 1e-2/np.sqrt(np.sqrt(200)),
                                   'arm': 1e-2/np.sqrt(np.sqrt(500)), '3darm': 1e-2/np.sqrt(np.sqrt(500))}
        self.knowledge['lr_as'] = {'ball': 9e-4/np.sqrt(np.sqrt(400)),
                                   'arm': 9e-4/np.sqrt(50), '3darm': 9e-4/np.sqrt(50)}

    def refresh(self, name=None):
        if name == 'mode':
            # mode determine a lot in policy
            tmp_mode = self.data['mode']
            self.get_data()
            self.data['mode'] = tmp_mode

            if self.data['mode'] == 'task_train':
                # action
                self.data['a_param_mark'] = ''
                self.data['a_activate'] = True
                self.data['a_names'] = self.upper_data['base']
                self.data['a_param_save'] = 'True'
                self.data['a_param_restore'] = 'False'
                self.data['a_trainable'] = 'True'
                self.data['lr_a'] = self.knowledge['lr_as'][self.upper_data['base']]
                # value
                self.data['c_param_mark'] = ''
                self.data['c_activate'] = True
                self.data['c_names'] = self.upper_data['base']
                self.data['c_param_save'] = 'True'
                self.data['c_param_restore'] = 'False'
                self.data['c_trainable'] = 'True'
                self.data['lr_c'] = self.knowledge['lr_cs'][self.upper_data['base']]

                # activation
                self.data['t_param_mark'] = ''
                self.data['t_activate'] = False

                # network
                self.data['network_mark'] = ''
                self.data['update_name'] = self.upper_data['base']
                self.data['bell_man'] = False
                # False?
                self.data['partial_restart'] = False

            elif self.data['mode'] == 'attribute_train':
                # attribute train only refers to train two layer caln
                # action
                self.data['a_param_mark'] = ''
                self.data['a_activate'] = True
                self.data['a_names'] = '{},{}'.format(self.upper_data['base'], self.upper_data['attribute_0'])
                self.data['a_param_save'] = 'False,True'
                self.data['a_param_restore'] = 'True,False'
                self.data['a_trainable'] = 'False,True'
                self.data['lr_a'] = self.knowledge['lr_as'][self.upper_data['base']]
                # value
                self.data['c_param_mark'] = ''
                self.data['c_activate'] = True
                self.data['c_names'] = '{},{}'.format(self.upper_data['base'], self.upper_data['attribute_0'])
                self.data['c_param_save'] = 'False,True'
                self.data['c_param_restore'] = 'True,False'
                self.data['c_trainable'] = 'False,True'
                self.data['lr_c'] = self.knowledge['lr_cs'][self.upper_data['base']]

                # activation
                self.data['t_param_mark'] = ''
                self.data['t_activate'] = False

                # network
                self.data['network_mark'] = ''
                self.data['update_name'] = self.upper_data['attribute_0']
                self.data['bell_man'] = False
                # False?
                self.data['partial_restart'] = False

            elif self.data['mode'] == 'weight_train':
                # attribute train only refers to train two layer caln
                # action
                self.data['a_param_mark'] = ''
                self.data['a_activate'] = True
                self.data['a_names'] = '{},{}'.format(self.upper_data['base'], self.upper_data['attribute_0'])
                self.data['a_param_save'] = 'False,False'
                self.data['a_param_restore'] = 'True,True'
                self.data['a_trainable'] = 'False,False'

                # value
                self.data['c_param_mark'] = ''
                self.data['c_activate'] = True
                self.data['c_names'] = '{},{}'.format(self.upper_data['base'], self.upper_data['attribute_0'])
                self.data['c_param_save'] = 'False,False'
                self.data['c_param_restore'] = 'True,True'
                self.data['c_trainable'] = 'False,False'

                # activation
                self.data['t_param_mark'] = ''
                self.data['t_activate'] = True
                self.data['t_names'] = self.upper_data['attribute_0']
                self.data['t_param_save'] = 'True'
                self.data['t_param_restore'] = 'False'
                self.data['t_trainable'] = 'True'
                self.data['lr_t'] = self.knowledge['lr_ts'][self.upper_data['base']]

                # network
                self.data['network_mark'] = ''
                self.data['update_name'] = self.upper_data['attribute_0']
                self.data['bell_man'] = False
                # False?
                self.data['partial_restart'] = False

                # data:
                self.data['random_level'] = 10.0
                self.data['update_string'] = 'stable'
                self.data['judge_string'] = 'stable'

            elif self.data['mode'] == 'test':
                # attribute train only refers to train two layer caln
                name_list = ''
                i = 0
                while ('attribute_{}'.format(i) in self.upper_data.keys()):
                    name_list += ',{}'.format(self.upper_data['attribute_{}'.format(i)])
                    i += 1

                # action
                self.data['a_param_mark'] = ''
                self.data['a_activate'] = True
                self.data['a_names'] = self.upper_data['base'] + name_list
                self.data['a_param_save'] = ('False,'*(self.upper_data['attribute_num'] + 1))[:-1]
                self.data['a_param_restore'] = ('True,'*(self.upper_data['attribute_num'] + 1))[:-1]
                self.data['a_trainable'] = ('False,'*(self.upper_data['attribute_num'] + 1))[:-1]

                # value
                self.data['c_param_mark'] = ''
                self.data['c_activate'] = False

                # activation
                self.data['t_param_mark'] = ''
                if int(self.upper_data['attribute_num']) > 1:
                    # only multi_attribute, we use t_param
                    self.data['t_activate'] = True
                    self.data['t_names'] = name_list[1:]
                    self.data['t_param_save'] = ('False,'*(self.upper_data['attribute_num']))[:-1]
                    self.data['t_param_restore'] = ('True,'*(self.upper_data['attribute_num']))[:-1]
                    self.data['t_trainable'] = ('False,'*(self.upper_data['attribute_num']))[:-1]
                    self.data['network_mark'] = ''
                    # if there is no trainable, update_name will not be used
                    self.data['update_name'] = None
                    self.data['bell_man'] = True
                else:
                    self.data['t_activate'] = False
                    self.data['network_mark'] = ''
                    # if there is no trainable, update_name will not be used
                    self.data['update_name'] = None
                    self.data['bell_man'] = False

                # False?
                self.data['partial_restart'] = False
        elif name == 'update_string':
            # if mode is weight_train, then the string can not be changed
            if self.data['mode'] == 'weight_train':
                print('can not choose update when weight train')
                raise ValueError

            self.data['judge_string'] = self.data['update_string']
            if (self.data['update_string'] == 'curriculum') or (self.data['update_string'] == 'reverse_curriculum'):
                self.data['random_level'] = 0.1
            else:
                self.data['random_level'] = 0.0

        elif name == 'judge_string':
            if self.data['mode'] == 'weight_train':
                print('can not choose judge when weight train')
                raise ValueError

            self.data['update_string'] = self.data['judge_string']
            if (self.data['update_string'] == 'curriculum') or (self.data['update_string'] == 'reverse_curriculum'):
                self.data['random_level'] = 0.1
            else:
                self.data['random_level'] = 0.0

    def push(self, updata):
        # rewrite push method for some data. which is needless to show
        self.data['global_step'] = 0
        self.data['threshold'] = 10.0
        # update list: which part will be update?
        if 'update_name' in self.data.keys():
            update_list = []
            if self.data['update_name'] != None:
                index = (self.upper_data['environment'].split(',')).index(self.data['update_name'])
                print('index is {}'.format(index))
                for part in ['a', 'c', 't']:
                    if part == 't':
                        index -= 1
                    if self.data['{}_activate'.format(part)]:
                        print(self.data['{}_trainable'.format(part)].split(',')[index])
                        if self.data['{}_trainable'.format(part)].split(',')[index] == 'True':
                            update_list.append(part)
            self.data['update_list'] = update_list

        # push
        for name, value in self.data.items():
            updata[name] = value


if __name__ == '__main__':
    policy_config('policy')
