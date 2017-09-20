# Author: Harvey Chang
# Email: chnme40cs@gmail.com
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import RL as rl
from RL.example.caln.policy_config import policy_config
from RL.example.caln.env_config import env_config
# environment:
from RL.env.mujoco.env import env


def main():
    whole_config = rl.configure.config()
    my_policy_config = policy_config('policy')
    my_env_config = env_config('environment')
    whole_config.add(my_env_config)
    whole_config.add(my_policy_config)
    rl.visual_tool.major_pane(whole_config)
    # get env:
    myenv = env(whole_config['environment'])



if __name__ == '__main__':
    main()