# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this file has run_episode, run_policy, one_start_run_episode, evaluate_start
import RL.utils.value_tools as V
import numpy as np


def run_episode(myconfig, myenv, mycontainer, mypolicy):
    # run episode means run and update for certain times;
    i = 0
    done = False

    # different type of reset:
    if myconfig['reset_from_pool']:
        data = mycontainer.reset(myenv.reset_from_pool())
    else:
        data = mycontainer.reset(myenv.reset(myconfig['random_level']))

    action = step(mypolicy, mycontainer, data)

    # run
    while ((i < myconfig['max_iter_num']) and not done):
        done, data = mycontainer.push(myenv.step(action))
        action = step(mypolicy, mycontainer, data)
        i += 1

    # data process: evaluate
    gamma = float(myconfig['gamma'])
    lam = float(myconfig['lam'])
    V.add_disc_sum_rew(mycontainer, gamma)
    V.add_gae(mycontainer, gamma, lam)
    # chose add sum_rewards, and, dis_sum_rewards:
    if myconfig['eval_string'] == 'sum':
        mycontainer['eval_value'] = [np.sum(mycontainer['rewards'])]
    elif myconfig['eval_string'] == 'dis_sum':
        mycontainer['eval_value'] = [np.mean(mycontainer['disc_sum_rew'])]
    else:
        print('No such eval_string')
        raise KeyError
    return mycontainer.pop()


def one_start_episode(myconfig, myenv, mycontainer, mypolicy, start_position):
    # run episode means run and update for certain times;
    i = 0
    done = False

    # reset from specific start:
    data = mycontainer.reset(myenv.inverse_set(start_position[0], start_position[1], start_position[2]))
    action = step(mypolicy, mycontainer, data)

    # run
    while ((i < myconfig['max_iter_num']) and not done):
        done, data = mycontainer.push(myenv.step(action))
        action = step(mypolicy, mycontainer, data)
        i += 1

    # data process: evaluate
    gamma = float(myconfig['gamma'])
    V.add_disc_sum_rew(mycontainer, gamma)
    # chose add sum_rewards, and, dis_sum_rewards:
    if myconfig['eval_string'] == 'sum':
        mycontainer['eval_value'] = [np.sum(mycontainer['rewards'])]
    elif myconfig['eval_string'] == 'dis_sum':
        mycontainer['eval_value'] = [np.mean(mycontainer['disc_sum_rew'])]
    else:
        print('No such eval_string')
        raise KeyError
    return mycontainer.pop()


def evaluate_start(myconfig, myenv, mycontainer, mypolicy, start_position):
    eval_value_list = []
    batch_epochs = int(myconfig['batch_epochs']/4)  # relatively low time of trying
    for i in range(batch_epochs):
        result = one_start_episode(myconfig=myconfig, myenv=myenv,
                             mycontainer=mycontainer, mypolicy=mypolicy, start_position=start_position)
        eval_value_list.append(result['eval_value'])
    return np.mean(eval_value_list)


def run_policy(myconfig, myenv, mycontainer, mypolicy):
    results = run_episode(myconfig=myconfig, myenv=myenv, mycontainer=mycontainer, mypolicy=mypolicy)
    del results['start_pos']

    # save results in a list:
    result_list = []
    batch_epochs = int(myconfig['batch_epochs'])
    for i in range(batch_epochs):
        result = run_episode(myconfig=myconfig, myenv=myenv, mycontainer=mycontainer, mypolicy=mypolicy)
        result_list.append(result)
    # filter when reset from random:
    all_passed = False
    if not myconfig['reset_from_pool']:
        # the only reason to filter is to find good starts:
        good_starts, all_passed = performance_filter(result_list, myconfig, myenv, mycontainer, mypolicy)

        if len(good_starts) != 0:
            myenv.set_start_pool(good_starts)
            myconfig['reset_from_pool'] = True

    for result in result_list:
        # we don't use start_pos more
        for key, value in result.items():
            if key != 'start_pos':
                results[key] = np.concatenate([results[key], value], axis=0)

    return results, all_passed


def step(mypolicy, mycontainer, data):
    action = mypolicy.predict_action(data)
    mycontainer['old_acts'].append(action)
    if mypolicy.policy_config['c_activate']:
        value = mypolicy.predict_value(data)
        mycontainer['values'].append([value])
    return action


# filter structure:
def performance_filter(results, myconfig, myenv, mycontainer, mypolicy):
    '''
    filter is used to filter bad performance.
    :param results: results: results is in the version of result list:
    :param threshold: the threshold to surpass
    :param ratio: ratio of choose
    :return: filtered results:
    '''
    filtered_starts = []
    threshold_low = float(myconfig['threshold_low'])
    threshold_high = float(myconfig['threshold_high'])
    all_zero = True
    all_passed = False

    for num, result in enumerate(results):
        average_performance = evaluate_start(myconfig, myenv, mycontainer, mypolicy, result['start_pos'])
        print('the {}\'s evaluation is {}'.format(num, average_performance))
        if (average_performance <= threshold_high) and (average_performance >= threshold_low):
            filtered_starts.append(result['start_pos'])
        if average_performance >= threshold_low:
            all_zero = False

    if (not filtered_starts) and (not all_zero):
        # means when all pass counter+1: try to avoid all 0:
        myconfig['counter'] += 1
        print('all passed, counter+1, counter is {}'.format(myconfig['counter']))
        all_passed = True

    return filtered_starts, all_passed   # can it be dealt with inside tensorflow?