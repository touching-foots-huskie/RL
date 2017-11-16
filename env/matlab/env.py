# : Harvey Chang
# : chnme40cs@gmail.com
# env_motor is the environment for matlab motor:
# function:
# start. run(weights, time)
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import traceback
import numpy as np
from RL.env.matlab.matlab_tool import *
from RL.utils.value_tools import square_error
import matlab.engine
import tqdm


class Env:
    def __init__(self, config, init_param_list):
        # what's important to init is to compel the related code
        self.env_config = config
        self.episode_num = config['episode_num']
        self.eng = matlab.engine.start_matlab()
        print('start engine!')
        # change work dir:
        self.eng.cd('/home/saturn/storage/codes/Harvey/RL/env/matlab')
        print('workspace have changed.')
        # first: initial array:
        # 3-layer network
        # run with zero weights
        self.input_dim = config['obs_dim']
        self.action_dim = config['act_dim']
        init_param_list = [matlab.double(m_array(param)) for param in init_param_list]
        self.init_param_list = init_param_list
        _, __, ___, error = self.run(self.init_param_list, self.env_config['total_len'])
        self.init_reward = square_error(self.env_config, error)
        print('initial finished!')

    def episode(self, param_list):
        # change type of param_list
        param_list = [matlab.double(m_array(param)) for param in param_list]
        results = []
        print('The Simulation episode start!')
        for i in tqdm.tqdm(range(self.episode_num)):
            result = dict()
            state, reward, action, error = self.run(param_list, self.env_config['total_len'])
            result['states'] = state
            result['rewards'] = square_error(self.env_config, error)  # -square error is the reward
            result['old_acts'] = action
            # the error is the eval value:
            result['eval_value'] = error
            # and value:
            try:
                if i == 0:
                    results = result
                else:
                    # merge together:
                    for key, value in result.items():
                        results[key] = np.concatenate([results[key], value], axis=0)
            except Exception:
                print(traceback.format_exc())

        return results

    def run(self, param_list, total_len):
        # run simulate:
        env_result = self.eng.simulate(*param_list)
        # divided it:
        env_result = np.array(env_result)
        env_state = env_result[:total_len, :7].reshape([-1, 7])
        env_reward = env_result[:total_len, 7].reshape([-1, 1])
        env_action = env_result[:total_len, 8].reshape([-1, 1])
        env_error = env_result[:total_len, 9].reshape([-1, 1])
        return env_state, env_reward, env_action, env_error


if __name__ == '__main__':
    # sample params
    pass

