# Author: Harvey Chang
# Email: chnme40cs@gmail.com
import RL as rl
import tensorflow as tf
from RL.example.caln.policy_config import PolicyConfig
from RL.example.caln.env_config import EnvConfig
# environment:
from RL.env.mujoco.env import env
from RL.env.mujoco.protocol import Container
# policy
from RL.nn.ppo import PolicyRep
from RL.train.judge import judge_wrapper
from RL.train.update import update_wrapper
from RL.train.evaluate import eval_wrapper
from RL.train.simulate import *
from RL.train.watch_dog import WatchDog
import RL.train.optim as optim
from collections import deque
import numpy as np


def update_policy(results, myconfig, mypolicy):
    for part in myconfig['update_list']:
        eval('optim.process_{}(results, myconfig, mypolicy)'.format(part))
    # record in the tensorboard: results is a variant factor
    mypolicy.log(results)


def main():
    # two global value for storing.
    GLOBAL_REWARDS = []
    GLOBAL_RANDOM_LEVEL = []
    # get config
    watch_dog = WatchDog()
    whole_config = rl.configure.configure.Config()
    my_policy_config = PolicyConfig('policy')
    my_env_config = EnvConfig('environment')
    whole_config.add(my_env_config)
    whole_config.add(my_policy_config)
    # visualization
    rl.configure.visual_tool.major_pane(whole_config)
    # get env:
    myenv = env(whole_config['environment'])
    mycontainer = Container(whole_config['attribute_num'])
    # get network:
    mypolicy = PolicyRep(whole_config)
    # init:
    mypolicy.sess.run(tf.global_variables_initializer())
    # flag:
    flag = False
    judge_func = judge_wrapper(whole_config)
    update_func = update_wrapper(whole_config)
    eval_func = eval_wrapper(whole_config)
    # long_term judge
    long_term_performance = deque([0.0]*whole_config['long_term_batch'],
                                  maxlen=whole_config['long_term_batch'])

    # restore:
    mypolicy.restore()
    if whole_config['mode'] != 'test':
        # train mode
        while not flag:
            results = run_policy(whole_config, myenv, mycontainer, mypolicy)
            eval_flag, average_performance = eval_func(results, whole_config, long_term_performance)
            # log average reward and random_level:
            GLOBAL_REWARDS.append(average_performance)
            GLOBAL_RANDOM_LEVEL.append(whole_config['random_level'])

            if eval_flag or whole_config['all_passed'] or whole_config['all_zero']:
                update_func(whole_config)
                long_term_performance = deque([0.0] * whole_config['long_term_batch'],
                                              maxlen=whole_config['long_term_batch'])
                watch_dog.refresh()
            # update param_meter:
            else:
                update_policy(results, whole_config, mypolicy)

            # watch dog start:
            if not watch_dog.check(average_performance):
                # if check false:
                if whole_config['random_level'] == 0.1:
                    # which means first start:
                    # mypolicy.restart_part()
                    mypolicy.refresh_sigma()
                else:
                    mypolicy.refresh_sigma()
                # refresh all:
                long_term_performance = deque([0.0] * whole_config['long_term_batch'],
                                              maxlen=whole_config['long_term_batch'])
                whole_config['reset_from_pool'] = False
                print('watch dog start, thread restart')

            whole_config['global_step'] += 1
            flag = judge_func(whole_config)

        # save in save mode.
        mypolicy.save()
        # save average performance & random level.
        GLOBAL_RANDOM_LEVEL = np.array(GLOBAL_RANDOM_LEVEL)
        GLOBAL_REWARDS = np.array(GLOBAL_REWARDS)
        np.save('{}/{}_reward.npy'.format(whole_config['perf_dir'], whole_config['environment']), GLOBAL_REWARDS)
        np.save('{}/{}_random.npy'.format(whole_config['perf_dir'], whole_config['environment']), GLOBAL_RANDOM_LEVEL)
        print('data_saved!')
    else:
        results = run_policy(whole_config, myenv, mycontainer, mypolicy)
        # make video: multi-attribute have not write.
        if whole_config['attribute_num']:
            add_state_list = [results['addition_{}'.format(i)] for i in range(whole_config['attribute_num'])]
            add_state_list = np.stack(add_state_list).transpose([1, 0, 2])
        else:
            add_state_list = []
        myenv.get_video(whole_config['environment'], results['states'], results['targets'], add_state_list)
        # print performance:
        print('the performance is {}'.format(np.mean(results['eval_value'])))
    print('Process end!')


if __name__ == '__main__':
    main()
