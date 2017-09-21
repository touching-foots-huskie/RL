# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this function is used to eval if start to eval:
import numpy as np


def eval_wrapper(config):
    eval_string = config['eval_string']
    if eval_string == 'sum':
        return sum_eval
    elif eval_string == 'dis_sum':
        return dis_sum_eval
    else:
        print('No such eval_string!')
        raise KeyError


def sum_eval(results, config):
    sum_flag = (np.mean(results['sum_rewards']) > config['reward_threshold'])
    return sum_flag


def dis_sum_eval(results, config):
    dis_sum_flag = (np.mean(results['dis_sum_rewards']) > config['reward_threshold'])
    return dis_sum_flag