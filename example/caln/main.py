# Author: Harvey Chang
# Email: chnme40cs@gmail.com
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import RL as rl
import tensorflow as tf
from RL.example.caln.policy_config import PolicyConfig
from RL.example.caln.env_config import EnvConfig
# environment:
from RL.env.mujoco.env import env
from RL.env.mujoco.protocol import Container
# policy
from RL.nn.ppo import PolicyRep
from RL.train.judge import judge_wrapper
from RL.train.update import update_wrapper
from RL.train.evaluate import eval_wrapper
from RL.train.simulate import *
from RL.train.watch_dog import WatchDog
import RL.train.optim as optim
from collections import deque


def update_policy(results, myconfig, mypolicy):
    for part in myconfig['update_list']:
        eval('optim.process_{}(results, myconfig, mypolicy)'.format(part))
    # record in the tensorboard: results is a variant factor
    mypolicy.log(results)


def main():
    # get config
    watch_dog = WatchDog()
    whole_config = rl.configure.configure.Config()
    my_policy_config = PolicyConfig('policy')
    my_env_config = EnvConfig('environment')
    whole_config.add(my_env_config)
    whole_config.add(my_policy_config)
    # visualization
    rl.configure.visual_tool.major_pane(whole_config)
    # get env:
    myenv = env(whole_config['environment'])
    mycontainer = Container(whole_config['attribute_num'])
    # get network:
    mypolicy = PolicyRep(whole_config)
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
        results = run_policy(whole_config, myenv, mycontainer, mypolicy)
        eval_flag, average_performance = eval_func(results, whole_config, long_term_performance)
        if eval_flag or whole_config['all_passed'] or whole_config['all_zero']:
            update_func(whole_config)
            long_term_performance = deque([0.0] * whole_config['long_term_batch'],
                                          maxlen=whole_config['long_term_batch'])
            watch_dog.refresh()
        # update param_meter:
        else:
            update_policy(results, whole_config, mypolicy)

        # watch dog start:
        if not watch_dog.check(average_performance):
            # if check false:
            if whole_config['random_level'] == 0.1:
                # which means first start:
                mypolicy.restart_part()
            else:
                mypolicy.refresh_sigma()
            whole_config['reset_from_pool'] = False
            print('watch dog start, thread restart')

        whole_config['global_step'] += 1
        flag = judge_func(whole_config)

    # save:
    mypolicy.save()
    print('Process end!')


if __name__ == '__main__':
    main()
