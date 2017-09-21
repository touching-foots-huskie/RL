# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this function is meant to update the config according to update_string:


def update_wrapper(config):
    update_string = config['update_string']
    if update_string in ['curriculum', 'reverse_curriculum']:
        return random_update
    elif update_string in ['stable']:
        return episode_update
    else:
        print('No such update_string!')
        raise KeyError


def random_update(config):
    config['random_level'] *= config['lambda']


def episode_update(config):
    pass