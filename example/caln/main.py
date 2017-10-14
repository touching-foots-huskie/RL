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
from RL.train.filter import filter
import RL.utils.value_tools as V
from RL.train import optim
import numpy as np
from collections import deque


def run_episode(myconfig, myenv, mycontainer, mypolicy):
    # run episode means run and update for certain times;
    i = 0
    done = False
    data = mycontainer.reset(myenv.reset(myconfig['random_level']))
    action = step(mypolicy, mycontainer, data)

    while ((i < myconfig['max_iter_num']) and not done):
        done, data = mycontainer.push(myenv.step(action))
        action = step(mypolicy, mycontainer, data)
        i += 1
    # add after done
    gamma = float(myconfig['gamma'])
    lam = float(myconfig['lam'])
    V.add_disc_sum_rew(mycontainer, gamma)
    V.add_gae(mycontainer, gamma, lam)
    # chose add sum_rewards, and, dis_sum_rewards:
    if myconfig['eval_string'] == 'sum':
        mycontainer['eval_value'] = [np.sum(mycontainer['rewards'])]
    elif myconfig['eval_string'] == 'dis_sum':
        mycontainer['eval_value'] = [np.sum(mycontainer['disc_sum_rew'])]
    else:
        print('No such eval_string')
        raise KeyError
    return mycontainer.pop()


def run_policy(myconfig, myenv, mycontainer, mypolicy):
    results = run_episode(myconfig=myconfig, myenv=myenv, mycontainer=mycontainer, mypolicy=mypolicy)
    # add disc_sum
    for i in range(int(myconfig['batch_epochs'])):
        result = run_episode(myconfig=myconfig, myenv=myenv, mycontainer=mycontainer, mypolicy=mypolicy)
        for key, value in result.items():
            results[key] = np.concatenate([results[key], value], axis=0)
    return results


def update_policy(results, myconfig, mypolicy):
    for part in myconfig['update_list']:
        eval('optim.process_{}(results, myconfig, mypolicy)'.format(part))
    # record in the tensorboard: results is a variant factor
    mypolicy.log(results)


def step(mypolicy, mycontainer, data):
    action = mypolicy.predict_action(data)
    mycontainer['old_acts'].append(action)
    if mypolicy.policy_config['c_activate']:
        value = mypolicy.predict_value(data)
        mycontainer['values'].append([value])
    return action


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
    # long_term judger
    long_term_performance = deque([0.0]*whole_config['long_term_batch'],
                                  maxlen=whole_config['long_term_batch'])

    while not flag:
        results = run_policy(whole_config, myenv, mycontainer, mypolicy)
        if eval_func(results, whole_config, long_term_performance):
            update_func(whole_config)
            long_term_performance = deque([0.0] * whole_config['long_term_batch'],
                                          maxlen=whole_config['long_term_batch'])
        # update param_meter:
        update_policy(results, whole_config, mypolicy)

        whole_config['global_step'] += 1
        flag = judge_func(whole_config)


if __name__ == '__main__':
    main()
