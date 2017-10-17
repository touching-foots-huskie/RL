# Author: Harvey Chang
# Email: chnme40cs@gmail.com
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import RL as rl
import tensorflow as tf
from RL.example.caln.policy_config import policy_config
from RL.example.caln.env_config import env_config
# environment:
from RL.env.mujoco.env import env
from RL.env.mujoco.protocol import container
# policy
from RL.nn.ppo import policy_rep
from RL.train.judge import judge_wrapper
from RL.train.update import update_wrapper
from RL.train.evaluate import eval_wrapper
from RL.train.simulate import *
import RL.train.optim as optim
from collections import deque


def update_policy(results, myconfig, mypolicy):
    for part in myconfig['update_list']:
        eval('optim.process_{}(results, myconfig, mypolicy)'.format(part))
    # record in the tensorboard: results is a variant factor
    mypolicy.log(results)


def main():
    # get config
    whole_config = rl.configure.configure.config()
    my_policy_config = policy_config('policy')
    my_env_config = env_config('environment')
    whole_config.add(my_env_config)
    whole_config.add(my_policy_config)
    # visualization
    rl.configure.visual_tool.major_pane(whole_config)
    # get env:
    myenv = env(whole_config['environment'])
    mycontainer = container(whole_config['attribute_num'])
    # get network:
    mypolicy = policy_rep(whole_config)
    # init:
    mypolicy.sess.run(tf.global_variables_initializer())
    # flag:
    flag = False
    judge_func = judge_wrapper(whole_config)
    update_func = update_wrapper(whole_config)
    eval_func = eval_wrapper(whole_config)
    # long_term judge
    long_term_performance = deque([0.0]*whole_config['long_term_batch'],
                                  maxlen=whole_config['long_term_batch'])

    # restore:
    mypolicy.restore()
    while not flag:
        results, all_passed = run_policy(whole_config, myenv, mycontainer, mypolicy)
        if eval_func(results, whole_config, long_term_performance) or all_passed:
            update_func(whole_config)
            long_term_performance = deque([0.0] * whole_config['long_term_batch'],
                                          maxlen=whole_config['long_term_batch'])
        # update param_meter:
        update_policy(results, whole_config, mypolicy)

        whole_config['global_step'] += 1
        flag = judge_func(whole_config)

    # save:
    mypolicy.save()
    print('Process end!')


if __name__ == '__main__':
    main()
