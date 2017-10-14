# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this function is used to eval if start to eval:
import numpy as np


def eval_wrapper(config):
    eval_string = config['eval_string']
    if (eval_string == 'sum') or (eval_string == 'dis_sum'):
        return sum_eval
    else:
        print('No such eval_string!')
        raise KeyError


def sum_eval(results, config, long_term_performance):
    performance = np.mean(results['eval_value'])
    long_term_performance.append(performance)
    average_performance = np.mean(long_term_performance)
    # delete the unused data
    del results['eval_value']
    sum_flag = (average_performance > config['reward_threshold'])
    print('episode: {}| eval_value: {}| random_level: {}'.format(config['global_step'],
                                                             average_performance,
                                                             config['random_level']))
    return sum_flag