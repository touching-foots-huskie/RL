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
    rewards = result['rewards']
    disc_sum_rew = discount(rewards, gamma)
    result['disc_sum_rew'] = disc_sum_rew


def add_gae(result, gamma, lam):
    rewards = np.array(result['rewards'])
    values = np.array(result['values'])
    # temporal differences
    try:
        if values.shape != ():
            tds = rewards - values + np.concatenate([values[1:] * gamma,[[0]]], axis=0)
        else:
            tds = rewards
    except Exception as e:
        print(e)
    advantages = discount(tds, gamma * lam)
    result['advantages'] = advantages
