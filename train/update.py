# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this function is meant to update the config according to update_string:


def update_wrapper(config):
    update_string = config['update_string']
    if update_string in ['curriculum', 'reverse_curriculum']:
        return random_update
    elif update_string in ['stable']:
        return episode_update
    elif update_string in ['posion_curriculum']:
        # you will get a counter:
        config['counter'] = 0
        return posion_update
    else:
        print('No such update_string!')
        raise KeyError


def random_update(config):
    config['random_level'] *= config['lambda']


def episode_update(config):
    pass


def posion_update(config):
    # another random start: total epoch_per_level
    config['reset_from_pool'] = False  # means start a new evaluation
    # random_level and max_iter change:
    if config['counter'] >= int(config['epoch_per_level']):
        config['random_level'] *= config['lambda']
        config['max_iter_num'] += 0.1
        config['counter'] = 0
    # dynamic change for threshold_low and threshold_high:
    if config['all_passed']:
        # increase high_threshold when all_passed
        config['actual_threshold_high'] = min(config['actual_threshold_high']+1, config['threshold_high'])
    else:
        # decrease when not all_passed
        config['actual_threshold_high'] = max(config['actual_threshold_high']-1, config['threshold_high_low'])
        
    if config['all_zero']:
        # increase high_threshold when all_zero
        config['actual_threshold_low'] = min(config['actual_threshold_low']+1, config['threshold_low_high'])
    else:
        # decrease when not all_zero
        config['actual_threshold_low'] = max(config['actual_threshold_low']-1, config['threshold_low'])