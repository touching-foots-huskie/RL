# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this function is used to judge if the main thread is going to stop according to
# judge string:


def judge_wrapper(config):
    judge_string = config['judge_string']
    if judge_string in ['curriculum', 'reverse_curriculum']:
        return random_judge
    elif judge_string in ['stable']:
        return episode_judge
    else:
        print('No such judge_string!')
        raise KeyError


def random_judge(config):
    random_flag = (config['random_level'] > config['threshold'])
    episode_flag = (config['global_step'] > config['total_episodes'])
    return random_flag or episode_flag


def episode_judge(config):
    episode_flag = (config['global_step'] > config['total_episodes'])
    return episode_flag