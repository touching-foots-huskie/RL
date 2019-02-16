# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# PPO for matlab
import tensorflow as tf
import RL.nn.matlab_net as pr


class PolicyRep(pr.PolicyRep):
    def __init__(self, policy_config):
        pr.PolicyRep.__init__(self, policy_config)
        # info:
        # using cpu:
        with tf.device('/cpu:0'):
            self.epsilon = float(self.policy_config.data['epsilon'])

            self.norm = tf.contrib.distributions.Normal(self.mean, self.sigma, name='normal')
            self.A = tf.squeeze(self.norm.sample(), name='action')

            # loss structure:
            '''
            # c_loss: we don't use c loss
            self.Adam_c = tf.train.AdamOptimizer(float(self.policy_config.data['lr_c']))
            with tf.name_scope('c_loss'):
                self.val_ph = tf.placeholder(tf.float32, (None, 1), 'val_valfunc')
                self.map_dict['rewards'] = self.val_ph
                self.c_loss = tf.reduce_mean(tf.square(self.value - self.val_ph), name='c_loss')
                self.c_update_op = self.Adam_c.minimize(self.c_loss)
                tf.summary.scalar('c_loss', self.c_loss)
            '''

            # a_loss
            self.Adam_a = tf.train.AdamOptimizer(float(self.policy_config.data['lr_a']))
            with tf.name_scope('a_loss'):
                with tf.name_scope('old_normal'):
                    self.old_sigma_ph = tf.placeholder(tf.float32, (None, self.action_dim,), 'old_sigma')
                    self.old_mean_ph = tf.placeholder(tf.float32, (None, self.action_dim), 'old_mean')
                    self.old_norm = tf.contrib.distributions.Normal(self.old_mean_ph, self.old_sigma_ph)

                    self.map_dict['old_means'] = self.old_mean_ph
                    self.map_dict['old_sigmas'] = self.old_sigma_ph

                with tf.name_scope('act_advantage'):
                    self.act_ph = tf.placeholder(tf.float32, (None, self.action_dim), 'act')
                    self.advantages_ph = tf.placeholder(tf.float32, (None, 1), 'advantages')  # (None, 1)

                    self.map_dict['advantages'] = self.advantages_ph
                    self.map_dict['old_acts'] = self.act_ph

                with tf.name_scope('r'):
                    self.r = tf.reduce_mean(self.norm.prob(self.act_ph) /
                                            (self.old_norm.prob(self.act_ph) + 1e-6), axis=-1)  # (None, 2)-> (None, 1)
                    self.r_clip = tf.clip_by_value(self.r, 1 - self.epsilon, 1 + self.epsilon)
                with tf.name_scope('J_clip'):
                    self.J = tf.minimum(self.r * self.advantages_ph, self.r_clip * self.advantages_ph)
                    self.a_loss = -tf.reduce_mean(self.J, name='a_loss')
                self.a_update_op = self.Adam_a.minimize(self.a_loss)
                tf.summary.scalar('a_loss', self.a_loss)

            # summary structure
            self.summary_op = tf.summary.merge_all()
            self.log_dir = self.policy_config.data['log_dir']
            self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    '''
    def update_c(self, data):
        feed_dict = self.get_feed_dict(data)
        self.sess.run(self.c_update_op, feed_dict)
    
    def update(self, data):
        # general update function
        for name in ['c', 'a']:
            eval('self.update_{}(data)'.format(name))

    '''

    def update_a(self, data):
        feed_dict = self.get_feed_dict(data)
        self.sess.run(self.a_update_op, feed_dict)

    def log(self, data):
        feed_dict = self.get_feed_dict(data)
        summary_str = self.sess.run(self.summary_op, feed_dict)
        self.summary_writer.add_summary(summary_str)




