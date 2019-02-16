# Author: Harvey Chang
# Email: chnme40cs@gmail.com
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import RL as rl
import tensorflow as tf
from RL.example.aug_irl.policy_config import PolicyConfig
from RL.example.aug_irl.env_config import EnvConfig
from RL.train.evaluate import eval_wrapper
from RL.train.judge import judge_wrapper
# environment:
from RL.env.matlab.env import Env

# policy
from RL.nn.matlab_ppo import PolicyRep
from collections import deque
import numpy as np
import RL.train.optim as optim


'''
def update_policy(results, myconfig, mypolicy):
    for part in ['c', 'a']:
        print('Train for {} part!'.format(part))
        eval('optim.us_process_{}(results, myconfig, mypolicy)'.format(part))
    # record in the tensorboard: results is a variant factor
    mypolicy.log(results)
'''


def update_policy(results, myconfig, mypolicy):
    optim.us_process_a(results, myconfig, mypolicy)
    mypolicy.log(results)


def run_policy(myconfig, myenv, mypolicy):
    # get results:
    results = myenv.episode(mypolicy.pop())
    # get value:
    results['values'] = mypolicy.predict_value(myconfig['amplify'])
    # results['advantages'] = (results['rewards'] - results['values']).reshape([-1, 1])
    rl.utils.value_tools.reward_compress(myconfig, results)
    # save the ad in the figure directory:
    rl.utils.figure_tool.result2np(results, ['advantages'])
    # refresh policy value:
    mypolicy.set_value(myconfig['amplify'], results['rewards'])
    # delete these short value| only the advantage is saved
    del results['rewards']
    del results['values']
    return results


if __name__ == '__main__':
    # get config
    whole_config = rl.configure.configure.Config()
    my_policy_config = PolicyConfig('policy')
    my_env_config = EnvConfig('environment')
    whole_config.add(my_env_config)
    whole_config.add(my_policy_config)
    # visualization
    rl.configure.visual_tool.major_pane(whole_config)
    # get network:
    mypolicy = PolicyRep(whole_config)
    mypolicy.init()
    print('policy init')
    # get env:
    myenv = Env(whole_config, mypolicy.pop())
    # set first value:
    mypolicy.set_value(whole_config['amplify'], np.repeat(myenv.init_reward, whole_config['episode_num'], axis=0))
    print('environment start!')
    eval_func = eval_wrapper(whole_config)
    judge_func = judge_wrapper(whole_config)
    long_term_performance = deque([0.0] * whole_config['long_term_batch'],
                                  maxlen=whole_config['long_term_batch'])

    # restore:
    if eval(whole_config['restore']):
        mypolicy.restore()

    # start run:
    while not judge_func(whole_config):
        results = run_policy(whole_config, myenv, mypolicy)
        eval_func(results, whole_config, long_term_performance)
        update_policy(results, whole_config, mypolicy)
        whole_config['global_step'] += 1

    if eval(whole_config['save']):
        mypolicy.save()
