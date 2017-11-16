# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# Value tools aims to manipulate the values/ discount, gae, and ...
# $File: value
# $Usage: Tool for value set:
import scipy.signal
import numpy as np


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(result, gamma):
    rewards = np.array(result['rewards']).ravel()
    disc_sum_rew = discount(rewards, gamma)
    result['disc_sum_rew'] = disc_sum_rew.reshape([-1, 1])


def add_gae(result, gamma, lam):
    rewards = np.array(result['rewards']).ravel()
    values = np.array(result['values']).ravel()
    # temporal differences
    try:
        if values.shape != ():
            tds = rewards - values + np.concatenate([values[1:] * gamma,[0]], axis=0)
        else:
            tds = rewards
    except Exception as e:
        print(e)
    advantages = discount(tds, gamma * lam)
    result['advantages'] = advantages.reshape([-1, 1])


def reward_compress(config, result):
    # reward compress is using point value to get advantage:
    total_len = config['total_len']*config['episode_num']
    section_len = config['section_len']  # length per section
    section_num = result['rewards'].shape[0]
    advantage = np.zeros([total_len, 1])

    for i in range(section_num):
        section_advantage = result['rewards'][i] - result['values'][i]
        # average advantage
        low_bound = int(i * section_len)
        high_bound = int((i + 1) * section_len)
        advantage[low_bound:high_bound] = section_advantage/section_len
    # save it in the results:
    result['advantages'] = advantage


def square_error(config, env_error):
    square_errors = np.zeros(config['section_num'])
    section_len = config['section_len']  # length per section
    for i in range(config['section_num']):
        # - is for bigger better:
        low_bound = int(i*section_len)
        high_bound = int((i+1)*section_len)
        square_errors[i] = -np.sum(np.square(env_error[low_bound:high_bound]))
    return square_errors
