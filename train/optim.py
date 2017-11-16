# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# $File: process.py
# $Usage: Interacte with action:
'''
Input State dicts with decrease reward
Output: feed_list, name_list, update_func, epochs,  batch_num 
'''
import numpy as np
from RL.train.shuffle_train import train


@train(True)
def process_a(results, myconfig, mypolicy):
    # state_dicts 
    epochs = int(myconfig['epochs'])
    batch_num = int(myconfig['batch_num'])
    update_func = mypolicy.update_a

    results['old_means'] = mypolicy.predict_mean(results)
    # length
    l = results['old_means'].shape[0]
    # tile the rest
    results['old_sigmas'] = np.tile(mypolicy.predict_sigma(results), [l, 1])
    # old_sigmas are 2 dimension, repeat it.
    return results, epochs, batch_num, update_func


@train(False)
def us_process_a(results, myconfig, mypolicy):
    # state_dicts
    epochs = int(myconfig['epochs'])
    batch_num = int(myconfig['batch_num'])
    update_func = mypolicy.update_a

    results['old_means'] = mypolicy.predict_mean(results)
    # length
    l = results['old_means'].shape[0]
    # tile the rest
    results['old_sigmas'] = np.tile(mypolicy.predict_sigma(results), [l, 1])
    # old_sigmas are 2 dimension, repeat it.
    return results, epochs, batch_num, update_func


@train(True)
def process_c(results, myconfig, mypolicy):
    # state_dicts
    epochs = int(myconfig['epochs'])
    batch_num = int(myconfig['batch_num'])
    update_func = mypolicy.update_c
    return results, epochs, batch_num, update_func


@train(False)
def us_process_c(results, myconfig, mypolicy):
    # state_dicts
    epochs = int(myconfig['epochs'])
    batch_num = int(myconfig['batch_num'])
    update_func = mypolicy.update_c
    return results, epochs, batch_num, update_func


@train(True)
def process_t(results, myconfig, mypolicy):
    # state_dicts
    epochs = int(myconfig['epochs'])
    batch_num = int(myconfig['batch_num'])
    update_func = mypolicy.update_t
    # nc_values:
    results['nc_values'] = results['disc_sum_rew']
    return results, epochs, batch_num, update_func