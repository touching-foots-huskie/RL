# Author: Harvey Chang
# Email: chnme40cs@gmail.com
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import RL as rl
import tensorflow as tf
from RL.example.caln.policy_config import policy_config
from RL.example.caln.env_config import env_config
# environment:
from RL.env.mujoco.env import env
from RL.env.mujoco.protocol import container
# policy
from RL.nn.ppo import policy_rep
from RL.train.judge import judge_wrapper

import numpy as np

def main():
    # get config
    whole_config = rl.configure.configure.config()
    my_policy_config = policy_config('policy')
    my_env_config = env_config('environment')
    whole_config.add(my_env_config)
    whole_config.add(my_policy_config)
    # visualization
    rl.configure.visual_tool.major_pane(whole_config)
    # get env:
    myenv = env(whole_config['environment'])
    mycontainer = container(whole_config['attribute_num'])
    # get network:
    mypolicy = policy_rep(whole_config)
    # init:
    mypolicy.sess.run(tf.global_variables_initializer())
    # flag:
    flag = True
    judge_func = judge_wrapper(whole_config)
    while not flag:
        results = run_policy(whole_config, myenv, mycontainer, mypolicy)

        flag = judge_func(whole_config)

def run_episode(myconfig, myenv, mycontainer, mypolicy):
    # run episode means run and update for certain times;
    i = 0
    done = False
    action_data = mycontainer.reset(myenv.reset(myconfig['random_level']))
    action = mypolicy.predict_action(action_data)
    mycontainer['actions'].append(action)
    while ((i < myconfig['max_iter_num']) and not done):
        done, action_data = mycontainer.push(myenv.step(action))
        action = mypolicy.predict_action(action_data)
        mycontainer['actions'].append(action)
        i += 1

    return mycontainer.pop()


def run_policy(myconfig, myenv, mycontainer, mypolicy):
    results = run_episode(myconfig=myconfig, myenv=myenv, mycontainer=mycontainer, mypolicy=mypolicy)
    for i in range(myconfig['batch_epochs']):
        result = run_episode(myconfig=myconfig, myenv=myenv, mycontainer=mycontainer, mypolicy=mypolicy)
        for key, value in result.items():
            results[key] = np.concatenate([results[key], value], axis=0)
    return results


if __name__ == '__main__':
    main()
