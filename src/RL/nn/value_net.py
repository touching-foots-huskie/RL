# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# Matlab net has special function for matlab performing
# Matlab net don't have the c network, we use value list to replace it.
# This network is used to predict the error of the motor to get an pre-ILC
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

        # save_path init:
        self.save_path = '{}_{}'.format(policy_config['save_dir'], self.policy_config['source'])
        with tf.device('/cpu:0'):
            # using cpu:
            with tf.name_scope('input'):
                layer = tf.placeholder('float32', [None, self.input_dim], 'input_tensor')
                self.input = layer
                self.map_dict['states'] = self.input

            # critic net:
            self.value = br.critic_net(self.input)

            # params:
            self.param_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor')

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            # get summary for all of the data:
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.summary.histogram(var.name, var)

            # c_loss: we only use c loss
            self.Adam_c = tf.train.AdamOptimizer(float(self.policy_config.data['lr_c']))
            with tf.name_scope('c_loss'):
                self.val_ph = tf.placeholder(tf.float32, (None, 1), 'val_valfunc')
                self.map_dict['rewards'] = self.val_ph
                self.c_loss = tf.reduce_mean(tf.square(self.value - self.val_ph), name='c_loss')
                self.c_update_op = self.Adam_c.minimize(self.c_loss)
                tf.summary.scalar('c_loss', self.c_loss)

            self.sess = tf.Session()
            # summary structure
            self.summary_op = tf.summary.merge_all()
            self.log_dir = self.policy_config.data['log_dir']
            self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

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

    def predict_value(self, data, name=None):
        feed_dict = self.get_feed_dict(data)
        return self.sess.run(self.value, feed_dict)

    def save(self):
        self.saver.save(self.sess, self.save_path)

    def restore(self):
        self.saver.restore(self.sess, self.save_path)

    def update_c(self, data):
        feed_dict = self.get_feed_dict(data)
        self.sess.run(self.c_update_op, feed_dict)

    def log(self, data):
        feed_dict = self.get_feed_dict(data)
        summary_str = self.sess.run(self.summary_op, feed_dict)
        self.summary_writer.add_summary(summary_str)


if __name__ == '__main__':
    pass