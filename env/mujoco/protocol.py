# $File protocol.py
# $Usage: detail process and store data structure.
import numpy as np

class container(object):
    def __init__(self, attribute_num):
        # using env_info for specific protocol setting
        self.attrib_num = attribute_num 

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def reset(self, *args):
        #
        state, target_state, add_state_list = args[0]
        self.data = {}
        self.data['states'] = [state]
        self.data['targets'] = [target_state]
        for i in range(self.attrib_num):
            self.data['addition_{}'.format(i)] = [add_state_list[i]]
        self.data['rewards'] = [[0.0]]
        self.data['old_acts'] = []
        self.data['values'] = []
        self.start_pos = args
        self.done = 0 # reset structure

        action_data = {}
        action_data['states'] = np.array([state])
        action_data['targets'] = np.array([target_state])
        for i in range(self.attrib_num):
            action_data['addition_{}'.format(i)] = np.array([add_state_list[i]])
        return action_data
    
    def push(self, *args):
        state, target_state, add_state_list, reward, done, punish_list = args[0]
        self.data['states'].append(state)
        self.data['targets'].append(target_state)
        for i in range(self.attrib_num):
            self.data['addition_{}'.format(i)].append(add_state_list[i])
        self.data['rewards'].append([reward])
        self.done = done
        # get action data
        action_data = {}
        action_data['states'] = np.array([state])
        action_data['targets'] = np.array([target_state])
        for i in range(self.attrib_num):
            action_data['addition_{}'.format(i)] = np.array([add_state_list[i]])

        return done, action_data

    def pop(self):
        # pop is pop out all of the data.// return list
        return self.data

    def get_start_state(self):
        # start_pos[0] can be used in inverse_set
        return self.start_pos[0]
        
