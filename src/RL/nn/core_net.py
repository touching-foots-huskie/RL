# $File: core_net.py
# $Author: Harvey Chang
import tensorflow as tf
import numpy as np


def actor_net(obs_ph, act_dim, suppress_ratio=1.0):
    with tf.variable_scope('actor'):
        # add batch_normalization to the data to replace scaler:
        obs_ph = tf.layers.batch_normalization(obs_ph)
        # placeholders:
        obs_dim = obs_ph.shape.as_list()[-1]  # the last dim of shape
        hid1_size = obs_dim * 10
        hid3_size = act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # hidden net:
        out = tf.layers.dense(obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio*np.sqrt(1/obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio*np.sqrt(1/hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio*np.sqrt(1/hid2_size)), name="h3")
        means = tf.layers.dense(out, act_dim, tf.tanh, kernel_initializer=tf.random_normal_initializer(
                                    stddev=suppress_ratio*np.sqrt(1 / hid3_size)), name='means')
        # variance:
        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, act_dim), tf.float32, tf.constant_initializer(0)) 
        sigma_init = tf.variables_initializer([log_vars], 'sigma_initializer')
        log_vars = tf.reduce_sum(log_vars, axis=0) - 1.0
        sigma = tf.exp(log_vars) 
        return means, sigma, sigma_init
      
        
def critic_net(obs_ph, suppress_ratio=1.0):
    with tf.variable_scope('critic'):
        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        obs_ph = tf.layers.batch_normalization(obs_ph)
        obs_dim = obs_ph.shape.as_list()[-1]  # the last dim of shape
        hid1_size = obs_dim * 10  # 10 chosen empirically on 'Hopper-v1'
        hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        # the learning rate is local rate.
        # 3 hidden layers with tanh activations
        out = tf.layers.dense(obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio * np.sqrt(1 / obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio * np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio * np.sqrt(1 / hid2_size)), name="h3")
        # freedom output of value
        out = tf.layers.dense(out, 1,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio * np.sqrt(1 / hid3_size)), name='output')
        # remove the dimension of 1
        out = tf.squeeze(out)
        return out


def activate_net(obs_ph):
    with tf.variable_scope('activate'):
        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        obs_ph = tf.layers.batch_normalization(obs_ph)
        obs_dim = obs_ph.shape.as_list()[-1]  # the last dim of shape
        hid1_size = obs_dim * 10  # 10 chosen empirically on 'Hopper-v1'
        hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        # the learning rate is local rate.
        # 3 hidden layers with tanh activations
        out = tf.layers.dense(obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3")
        # freedom output of value
        out = tf.layers.dense(out, 1,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid3_size)), name='output')
        # remove the dimension of 1
        out = tf.squeeze(out)
        return out
