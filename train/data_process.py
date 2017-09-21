# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# $File: process.py
# $Usage: Interacte with action:
'''
Input State dicts with decrease reward
Output: feed_list, name_list, update_func, epochs,  batch_num
'''
import System.Update.Train.batch_train.train as train
import System.Utils.value as V


@train
def process_a(state_dicts, abstract_policy):
    # state_dicts
    epochs = abstract_policy.policy_config['epochs']
    batch_num = abstract_policy.policy_config['batch_num']
    update_config = ['a']

    gamma = abstract_policy.policy_config['gamma']
    lamda = abstract_policy.policy_config['lamda']
    for s in state_dicts:
        feed_dict, value = abstract_policy.get_feed_dict(state_dicts, True)
        s['values'] = value
        s['advantage'] = V.add_gae(s, gamma, lamda)

    m_states = V.merge_dicts(state_dicts)
    feed_dict = abstract_policy.get_feed_dict(m_states)
    m_states['old_mean'] = abstract_policy.nf.predict_mean(feed_dict)
    m_states['old_sigma'] = abstract_policy.nf.predict_sigma(feed_dict)
    m_states['old_act'] = abstract_policy.nf.predict_action(feed_dict)

    feed_dict = abstract_policy.get_feed_dict(m_states)
    return feed_dict, epochs, batch_num, update_config, abstract_policy


@train
def process_c(state_dicts, abstract_policy):
    # state_dicts
    epochs = abstract_policy.policy_config['epochs']
    batch_num = abstract_policy.policy_config['batch_num']
    update_config = ['c']

    gamma = abstract_policy.policy_config['gamma']
    for s in state_dicts:
        s['dis_sum_rewards'] = V.disc_sum_rew(s, gamma)

    m_states = V.merge_dicts(state_dicts)
    feed_dict = abstract_policy.get_feed_dict(m_states)
    return feed_dict, epochs, batch_num, update_config, abstract_policy


@train
def process_t(state_dicts, abstract_policy):
    # state_dicts
    epochs = abstract_policy.policy_config['epochs']
    batch_num = abstract_policy.policy_config['batch_num']
    update_config = ['c']

    gamma = abstract_policy.policy_config['gamma']
    # in t update, nc_values is the disc_sum_reward
    for s in state_dicts:
        s['nc_values'] = V.disc_sum_rew(s, gamma)

    m_states = V.merge_dicts(state_dicts)
    feed_dict = abstract_policy.get_feed_dict(m_states)
    return feed_dict, epochs, batch_num, update_config, abstract_policy

