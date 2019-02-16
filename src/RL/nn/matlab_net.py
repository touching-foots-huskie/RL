# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# Matlab net has special function for matlab performing
# Matlab net don't have the c network, we use value list to replace it.
# : Harvey Chang
# : chnme40cs@gmail.com
import tensorflow as tf
import numpy as np
import RL.nn.matlab_core_net as br
# use these to test


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
        self.sect_num = policy_config['section_num']

        # value is saved in a dict of many lists for different amplitude.
        self.value_dict = dict()

        # save_path init:
        self.save_path = '{}_{}'.format(policy_config['save_dir'], self.policy_config['source'])
        with tf.device('/cpu:0'):
            # using cpu:
            with tf.name_scope('input'):
                layer = tf.placeholder('float32', [None, self.input_dim], 'input_tensor')
                self.input = layer
                self.map_dict['states'] = self.input

            # action net:
            self.mean, self.sigma, self.sigma_init = br.actor_net(self.input, self.action_dim)
            # critic net:

            # params:
            self.param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor')

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            # get summary for all of the data:
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.summary.histogram(var.name, var)

            self.sess = tf.Session()

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def pop(self):
        # pop out a param_list
        return self.sess.run(self.param_list)

    def get_feed_dict(self, data):
        feed_dict = {}
        for name, d in data.items():
            if name in self.map_dict.keys():
                feed_dict[self.map_dict[name]] = d
        return feed_dict

    # action
    def predict_mean(self, data, name=None):
        feed_dict = self.get_feed_dict(data)
        return self.sess.run(self.mean, feed_dict)

    def predict_sigma(self, data, name=None):
        feed_dict = self.get_feed_dict(data)
        return self.sess.run(self.sigma, feed_dict)

    # value for matlab:
    def predict_value(self, amplitude):
        # value dict has a list for different amplitude
        # return a list, leave it to be processed by the value_tool
        return self.value_dict[amplitude]

    def set_value(self, amplitude, performance_list):
        # performance_list is given by the value_tool
        # assert len(performance_list) == self.sect_num
        self.value_dict[amplitude] = performance_list

    def save(self):
        self.saver.save(self.sess, self.save_path)

    def restore(self):
        self.saver.restore(self.sess, self.save_path)


if __name__ == '__main__':
    pass