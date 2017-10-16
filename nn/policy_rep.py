# $File: policy_rep: policy representation
# $Usage: This file is used to establish the abstract net for attribute
# In order to decrease the covariation of net, we will save those model in Num:
# Each config will be saved in the dict.json file.

import tensorflow as tf
import RL.nn.write_dict as write_dict
import RL.nn.core_net as br 


class policy_rep(object):
    def __init__(self, policy_config):
        '''
        policy_config is the config: policy config will not change in process
        Name_list: 
        Dim_list: (state, addition_state_0...)
        '''
        # index:
        if policy_config['update_name'] != None:
            self.index = (policy_config['environment'].split(',')).index(policy_config['update_name'])
        else:
            self.index = 0
        # connect in two world
        self.map_dict = {}
        self.sess = tf.Session()
        self.policy_config = policy_config

        # init save place:
        self.save_path_dict = write_dict.write_dict() 
        for name in self.save_path_dict.keys():
            self.save_path_dict[name] = self.policy_config.data['save_dir'] + self.save_path_dict[name]

        # infos:
        self.update_name = self.policy_config.data['update_name']
        self.base_name = self.policy_config.data['a_names'].split(',')[0]
        self.attrib_num = self.policy_config.data['attribute_num']

        # init states
        self.s = []
        for dim_num in self.policy_config.data['dim_list'].split(','): 
            s = tf.placeholder(tf.float32, [None, int(dim_num)])
            self.s.append(s)
        # mappings:
        self.map_dict['states'] = self.s[0]
        self.map_dict['targets'] = self.s[1]
        for i in range(self.attrib_num):
            self.map_dict['addition_{}'.format(i)] = self.s[i + 2]

        self.action_dim = self.policy_config.data['action_dim']
        # CLAN self.mean, self.sigma are those for output:
        self.mean = 0
        self.sigma = 1e-4

        self.means = {}
        self.sigmas = {}
        self.sigma_inits = {}
        self.saver_dict = {}
        self.param_dict = {}
        for i, name in enumerate(self.policy_config.data['a_names'].split(',')):
            with tf.variable_scope(name) as scope:
                # if reuse attribute
                if name in self.policy_config.data['a_names'].split(',')[:i]:
                    scope.reuse_variables()
                if i == 0:
                    cs = tf.concat([self.s[0], self.s[1]], axis = -1)
                    mean, sigma, sigma_init = br.actor_net(cs, self.action_dim)
                else:
                    cs = tf.concat([self.s[0], self.s[i], mean], axis = -1)
                    mean, sigma, sigma_init = br.actor_net(cs, self.action_dim)

                self.sigma_inits[name] = sigma_init
                self.means[name] = mean
                self.sigmas[name] = sigma
                self.mean = mean + self.mean
                self.param_dict['{}_action'.format(name)] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'{}/{}'.format(name,'actor'))
                self.saver_dict['{}_action'.format(name)] = tf.train.Saver(self.param_dict['{}_action'.format(name)])
        # choose sigma:
        if self.update_name != None:
            self.sigma = self.sigmas[self.update_name]
                
        if self.policy_config.data['c_activate']:
            # value is the average reward with policy
            self.value = 0
            self.values = {}
            for i, name in enumerate(self.policy_config.data['c_names'].split(',')):
                with tf.variable_scope(name) as scope:
                    if name in self.policy_config.data['c_names'].split(',')[:i]:
                        scope.reuse_variables()
                    if i == 0:
                        cs = tf.concat([self.s[0], self.s[1]], axis = -1)
                        value = br.critic_net(cs)
                    else:
                        # we don't need to concern with mean, which is too detail to converge
                        cs = tf.concat([self.s[0], self.s[i]], axis = -1)
                        value = br.critic_net(cs)

                    self.values[name] = value
                    self.value = value + self.value
                    self.param_dict['{}_value'.format(name)] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'{}/{}'.format(name,'critic'))
                    self.saver_dict['{}_value'.format(name)] = tf.train.Saver(self.param_dict['{}_value'.format(name)])
                 
        if self.policy_config.data['t_activate']:
            # activation is the average reward without policy
            # use activation and train activation is different
            # activation = pre_vlaue - {no policy reward} # means the importance of target policy
            self.activations = {}
            for i, name in enumerate(self.policy_config.data['t_names'].split(',')):
                with tf.variable_scope(name) as scope:
                    if name in self.policy_config.data['t_names'].split(',')[:i]:
                        scope.reuse_variables()
                    # activation is invariant to mean before
                    cs = tf.concat([self.s[0], self.s[i]], axis = -1)
                    activation = br.activate_net(cs)

                    self.activations[name] = activation
                    self.param_dict['{}_activation'.format(name)] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'{}/{}'.format(name,'activate'))
                    self.saver_dict['{}_activation'.format(name)] = tf.train.Saver(self.param_dict['{}_value'.format(name)])
        
            # bell_man structure:
            if self.policy_config.data['bell_man']:
                self.bell_man_weights()

        if self.policy_config.data['partial_restart']:
            self.init_dicts = {} 
            for name, variables in self.param_dict.items():
                self.init_dicts[name] = tf.variables_initializer(variables)

    def bell_man_weights(self):
        '''
        Reweight the action according to bellman combination
        '''
        self.ks = {}
        sum_ks = 0.0
        for name in self.policy_config.data['t_names'].split(','):
            self.ks[name] = tf.maximum(self.activations[name], 0.0)
            sum_ks += self.ks[name]
        self.flag = tf.cond(tf.greater(sum_ks, 1e-4), lambda: 1.0, lambda: 0.0)
        sum_ks += 1e-4
        # get new self.mean
        self.mean = self.means[self.base_name]
        for name in self.policy_config.data['t_names'].split(','):
            self.means[name] = (self.means[name] * self.ks[name] / sum_ks)*self.flag
            self.mean += self.means[name]

    def save(self):
        # get param_save:
        self.policy_config.data['param_save'] = {}
        # a:
        for i, value in enumerate(self.policy_config['a_param_save'].split(',')):
            self.policy_config.data['param_save']['{}_action'.format(self.policy_config['a_names'].split(',')[i])] = True if value == 'True' else False
        # c
        if self.policy_config.data['c_activate']:
            for i, value in enumerate(self.policy_config['c_param_save'].split(',')):
                self.policy_config.data['param_save']['{}_value'.format(self.policy_config['c_names'].split(',')[i])] = True if value == 'True' else False

        # t
        if self.policy_config.data['t_activate']:
            for i, value in enumerate(self.policy_config['t_param_save'].split(',')):
                self.policy_config.data['param_save']['{}_activation'.format(self.policy_config['t_names'].split(',')[i])] = True if value == 'True' else False

        for name in self.param_dict.keys():
            if self.policy_config.data['param_save'][name]:
                self.saver_dict[name].save(self.sess,
                                           self.save_path_dict['{}_{}'.format(self.base_name, name)])

    def restore(self):
        self.policy_config.data['param_restore'] = {}
        # a:
        for i, value in enumerate(self.policy_config['a_param_restore'].split(',')):
            self.policy_config.data['param_restore'][
                '{}_action'.format(self.policy_config['a_names'].split(',')[i])] = True if value == 'True' else False
        # c
        if self.policy_config.data['c_activate']:
            for i, value in enumerate(self.policy_config['c_param_restore'].split(',')):
                self.policy_config.data['param_restore'][
                    '{}_value'.format(self.policy_config['c_names'].split(',')[i])] = True if value == 'True' else False

        # t
        if self.policy_config.data['t_activate']:
            for i, value in enumerate(self.policy_config['t_param_restore'].split(',')):
                self.policy_config.data['param_restore']['{}_activation'.format(
                    self.policy_config['t_names'].split(',')[i])] = True if value == 'True' else False

        for name in self.param_dict.keys():
            if self.policy_config.data['param_restore'][name]:
                self.saver_dict[name].restore(self.sess,
                                              self.save_path_dict['{}_{}'.format(self.base_name, name)])

    def restart_part(self, name):
        self.sess.run(self.init_dicts[name])         

    def refresh_sigma(self, name):
        self.sess.run(self.sigma_inits[name])
     
    def get_feed_dict(self, data):
        '''
        Change state dict into feed of state
        feed_config = [typ: clan/tradition, cal_reward]
        '''
        feed_dict = {}
        for name, d in data.items():
            if name in self.map_dict.keys():
                feed_dict[self.map_dict[name]] = d
        return feed_dict

    def predict_mean(self, data, name=None):
        '''
        Args: states
        '''
        feed_dict = self.get_feed_dict(data)
        if name != None:
            return self.sess.run(self.means[name], feed_dict)
        else:
            return self.sess.run(self.mean, feed_dict)

    def predict_sigma(self, data, name=None):
        '''
        Args: states
        '''
        feed_dict = self.get_feed_dict(data)
        if name != None:
            return self.sess.run(self.sigmas[name], feed_dict)
        else:
            return self.sess.run(self.sigma, feed_dict)

    def predict_value(self, data, name=None):
        '''
        Args: states
        '''
        feed_dict = self.get_feed_dict(data)
        if name != None:
            return self.sess.run(self.values[name], feed_dict)
        else:
            return self.sess.run(self.value, feed_dict)


