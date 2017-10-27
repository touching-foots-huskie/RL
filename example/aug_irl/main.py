# Author: Harvey Chang
# Email: chnme40cs@gmail.com
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import RL as rl
import tensorflow as tf
from RL.example.aug_irl.policy_config import PolicyConfig
from RL.example.aug_irl.env_config import EnvConfig

# environment:
from RL.env.matlab.env import Env

# policy
from collections import deque
import numpy as np


def update_policy(results, myconfig, mypolicy):
    pass


def main():
    # get config
    whole_config = rl.configure.configure.Config()
    my_policy_config = PolicyConfig('policy')
    my_env_config = EnvConfig('environment')
    whole_config.add(my_env_config)
    whole_config.add(my_policy_config)
    # visualization
    rl.configure.visual_tool.major_pane(whole_config)
    # get env:
    myenv = Env(whole_config)
    print('environment start!')
    # get network:


if __name__ == '__main__':
    main()
