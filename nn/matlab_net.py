# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# Matlab net has special function for matlab performing
# : Harvey Chang
# : chnme40cs@gmail.com
import tensorflow as tf
import numpy as np
import RL.nn.matlab_core_net as br


class PolicyRep:
    def __init__(self, policy_config):
        # layers setting:
        self.policy_config = policy_config
        self.param_dict = dict()
        self.saver_dict = dict()
        self.map_dict = dict()
        self.save_path_dict = dict()
        self.input_dim = policy_config['obs_dim']
        self.action_dim = policy_config['act_dim']

        # save_path init:
        self.save_path = '{}_{}'.format(policy_config['save_dir'], self.policy_config['source'])

        with tf.name_scope('input'):
            layer = tf.placeholder('float32', [None, self.input_dim], 'input_tensor')
            self.input = layer
            self.map_dict['state'] = self.input

        # action net:
        self.mean, self.sigma, self.sigma_init = br.actor_net(self.input, self.action_dim)
        # critic net:
        self.value = br.critic_net(self.input)

        # params:
        '''
        self.param_dict['action'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor')
        self.param_dict['value'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic')
        self.saver_dict['action'] = tf.train.Saver(self.param_dict['action'])
        self.saver_dict['value'] = tf.train.Saver(self.param_dict['value'])
        '''
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        self.sess = tf.Session()

    def pop(self):
        # pop out a param_list
        pass

    def save(self):
        self.saver.save(self.sess, self.save_path)

    def restore(self):
        self.saver.restore(self.sess, self.save_path)


if __name__ == '__main__':
    pass

