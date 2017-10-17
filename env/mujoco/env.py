# $File: environment.py
# $Author: Harvey Chang
'''
env is the force version of environment
'''
import sys
sys.path.append('/mnt/storage/codes/Harvey')

import os
import cv2
import math
import random
import traceback
import numpy as np
from RL.env.mujoco import robot


image_size = 200
path = '/mnt/storage/codes/Harvey/RL/env/mujoco'

'''
task_name:
    ball: ball 
    arm:  arm robot
    3darm: 3 dimension arm
    None: direct train attribute
attribute_name:
    safety:obstacle to avoid
    door:  a time will open at a certain time
    irl:   a walking obstacle
    ball_safety: ball in safety
jnt:
    state, target, add1, add2, add3
force:
    state, add
'''
class env:
    def __init__(self, environment):
        '''
        :param environment: environment looks like:
        '''
        name_list = environment.split(',')
        task_name = name_list[0]
        attribute_name = ''
        for name in name_list[1:]:
            attribute_name += '_{}'.format(name)
        if attribute_name != '':
            attribute_name = attribute_name[1:]

        self.task_name = task_name
        self.attribute_name = attribute_name
        self.default_action = self.step_walk
        self.default_add_action = self.step_add_walk
        self.enable_ad = False
        self.accum_f = lambda a, b: sum(b[:a])
        # the last position in state_num is a flag for collision
        if task_name == 'DA':
            # this is the special attribute_env:
            if attribute_name == 'ball_safety':
                self.attribute_list = ['ball_safety']
                filename = '{}/frames/ball_obstacle.xml'.format(path)
                self.rp_list = [0.0, -0.3]
                self.state_num = 2  # state_num is the number of state dimension of itself.
                self.target_num = 0 # target_num is the number of target dimension of itself. In attribute_structure, there is no place for it.
                self.add_state_num_list = [2]
                self.add_action_num_list = [2]
                self.init_qpos =[[-0.1, -0.1, 0.1, 0.1],
                                  [0.1, 0.1, -0.1, -0.1],
                                  [-0.1, 0.1, 0.1, -0.1],
                                  [0.1, -0.1, -0.1, 0.1],
                                 ]

        elif task_name == 'ball':
            self.state_num = 2  # state_num is the number of state dimension of itself.
            self.target_num = 2 # target_num is the number of target dimension of itself. In attribute_structure, there is no place for it.
            if attribute_name == '':
                self.attribute_list = []
                filename = '{}/frames/ball.xml'.format(path)
                self.rp_list = [1.0, -0.3]
                self.add_state_num_list = []
                self.add_action_num_list = []
                self.init_qpos =[[-0.1, -0.1, 0.1, 0.1],
                                 [0.1, 0.1, -0.1, -0.1],
                                 ]
                # self.init_qpos =[[-0.1, -0.1, 0.1, 0.1]]

            elif attribute_name == 'ball_safety':
                self.attribute_list = ['ball_safety']
                filename = '{}/frames/obstacle_ball.xml'.format(path)
                self.rp_list = [1.0, -0.01]
                self.add_state_num_list = [2]
                self.add_action_num_list = [2]
                # common version:
                # self.init_qpos =[[-0.3, -0.3, 0.3, 0.3, 0.0, 0.0],
                #                 ]
                # reverse version:
                self.init_qpos =[[0.17, 0.17, 0.3, 0.3, 0.0, 0.0],]
             
            elif attribute_name == 'safety':
                self.attribute_list = ['safety']
                filename = '{}/frames/obstacle_ball.xml'.format(path)
                self.rp_list = [1.0, -0.01]
                self.add_state_num_list = [2]
                self.add_action_num_list = [2]
                # common version
                # self.init_qpos =[[0.17, 0.17, 0.3, 0.3, 0.0, 0.0],]
                # reverse version:
                # self.init_qpos =[[0.17, -0.17, 0.3, 0.3, 0.0, 0.0],
                #                 [-0.17, 0.17, 0.3, 0.3, 0.0, 0.0],
                #                ]
                self.init_qpos =[[-0.3, -0.3, 0.3, 0.3, 0.0, 0.0],
                                [0.3, 0.3, -0.3, -0.3, 0.0, 0.0],
                                ]

                # [-0.3, 0.3, 0.3, -0.3, 0.0, 0.0],
                # [-0.17, 0.17, 0.3, 0.3, 0.0, 0.0],
                # [0.17, -0.17, 0.3, 0.3, 0.0, 0.0],

            elif attribute_name == 'safety_safety':
                self.attribute_list = ['safety', 'safety']
                filename = '{}/frames/obstacle_obstacle_ball.xml'.format(path)
                self.rp_list = [1.0, -0.01, -0.01]
                self.add_state_num_list = [2, 2]
                self.add_action_num_list = [2, 2]
                # common version
                # self.init_qpos =[[-0.3, -0.3, 0.3, 0.3, 0.0, 0.0],
                #                [0.3, 0.3, -0.3, -0.3, 0.0, 0.0],
                #                ]

                # reverse version:
                self.init_qpos =[[-0.4, -0.4, 0.4, 0.4, -0.15, -0.15, 0.15, 0.15],
                                [0.4, 0.4, -0.4, -0.4, -0.15, -0.15, 0.15, 0.15],
                                ]

            elif attribute_name == 'door':
                # strengthen the door
                self.attribute_list = ['door']
                filename = '{}/frames/door_ball.xml'.format(path)
                self.rp_list = [1.0, -0.01]
                self.add_state_num_list = [1]
                self.add_action_num_list = [1]
                # common version
                self.init_qpos =[[-0.3, -0.3, 0.3, 0.3, 0.0],
                                ]
                # [0.3, 0.3, -0.3, -0.3, 0.0],
                self.step_num = 10 # step num is the time to open door decrease the time
                self.door_num = 0 # door is in the first position

            elif attribute_name == 'traffic':
                self.attribute_list = ['traffic']
                filename = '{}/frames/traffic_ball.xml'.format(path)
                self.rp_list = [1.0, -0.1]
                self.add_state_num_list = [1]
                self.add_action_num_list = [1]
                # common version
                self.init_qpos =[[-0.3, -0.3, 0.3, 0.3, 0.5],
                                [0.3, 0.3, -0.3, -0.3, 0.5],
                                ]
                # [0.3, 0.3, -0.3, -0.3, 0.0],
                self.traffic_interval = 4 # step num is the time to open door decrease the time
                self.traffic_num = 0
            
            elif attribute_name == 'safety_traffic':
                self.attribute_list = ['safety', 'traffic']
                filename = '{}/frames/safety_traffic_ball.xml'.format(path)
                self.rp_list = [1.0, -0.01, -1.0]
                self.add_state_num_list = [2, 1]
                self.add_action_num_list = [2, 1]
                # common version
                self.init_qpos =[[-0.3, -0.3, 0.3, 0.3, 0.0, 0.0, 0.5],
                                ]
                # [0.3, 0.3, -0.3, -0.3, 0.0],
                self.traffic_interval = 4 # step num is the time to open door decrease the time
                self.traffic_num = 1

            else:
                print('{} doesn\' t have {}'.format(task_name, attribute_name))
                traceback.print_exc()

        elif task_name == 'arm':
            self.state_num = 5
            self.target_num = 2

            if attribute_name == '':
                self.attribute_list = []
                filename = '{}/frames/arm.xml'.format(path)
                self.rp_list = [1.0, -0.3]
                self.add_state_num_list = []
                self.add_action_num_list = []
                self.init_qpos =[[-0.5, -0.2, 0.0, 0.0, 0.0, 0.3, 0.0],
                                [-0.7, -0.2, 0.0, 0.0, 0.0, 0.3, 0.0],
                                ]

            elif attribute_name == 'safety':
                self.attribute_list = ['safety']
                filename = '{}/frames/obstacle_arm.xml'.format(path)
                self.rp_list = [1.0, -0.01]
                self.add_state_num_list = [2]
                self.add_action_num_list = [2]
                self.init_qpos =[[-0.5, -1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                                 [-0.5, 1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                                ]

            elif attribute_name == 'door':
                self.attribute_list = ['door']
                filename = '{}/frames/door_arm.xml'.format(path)
                self.rp_list = [1.0, -0.01]
                self.add_state_num_list = [1]
                self.add_action_num_list = [1]
                # common version
                self.init_qpos = [[-0.5, -1.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0],
                                  ]
                self.step_num = 10  # step num is the time to open door decrease the time
                self.door_num = 0  # door is in the first position

            else:
                print('{} doesn\' t have {}'.format(task_name, attribute_name))
                traceback.print_exc()

        elif task_name == '3darm':
            self.state_num = 5
            self.target_num = 3
            if attribute_name == '':
                self.attribute_list = []
                filename = '{}/frames/3darm.xml'.format(path)
                self.rp_list = [1.0, -0.3]
                self.add_state_num_list = []
                self.add_action_num_list = []
                self.init_qpos =[[0.06, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                               #  [0.06, 0.12, 0.0, 0.0, 0.0, 0.0, -0.1, -0.1]
            else:
                print('{} doesn\' t have {}'.format(task_name, attribute_name))
        else:
            print('we don\'t have {}'.format(task_name))
            traceback.print_exc()

        self.acc_add_state_num_list = [self.accum_f(i, self.add_state_num_list) for i  in range(len(self.add_state_num_list) + 1)]
        self.acc_add_action_num_list = [self.accum_f(i, self.add_action_num_list) for i  in range(len(self.add_action_num_list) + 1)]
        self.rb = robot.robot(filename, 100) 
        self.init_state = self.rb.sim.get_state()
        self.get_scaler()
        self.get_range_scaler()
        # adding time property
        self.time = 0.0
        # additstep_walk:
        self.direction = 1.0
                
    def choose_init_qpos(self):
        # randomly choose init_qpos:
        numbers = len(self.init_qpos)
        num = random.randint(0, numbers - 1)
        return self.init_qpos[num]

    def reset(self, random_scale = 0.0):
        # real reset
        '''
        random_scale should between 0~10
        '''
        random_scale = np.clip(random_scale, 0.0, 10.0)
        self.rb.sim.set_state(self.init_state)
        self.time = 0.0
        self.direction = 1.0

        # clear the force on object:
        self.step_walk([0.0] * self.state_num)
        # clear the force on obstcle
        for i, l in enumerate(self.add_action_num_list):
            self.step_add_walk([0.0]*l, i)

        # random init all of the data on target and state and add_state
        self.get_range_scaler()
        # reset the scaler
        for i in range(len(self.range_scaler)):
            low_bound, high_bound, center, scale = self.range_scaler[i]
            offset = (1 - random_scale/10.0) * (center - low_bound) + low_bound
            # cirucum learning
            self.rb.sim.data.qpos[i] = random.random() * random_scale * scale + offset  
        # if not proper set
        # simulate for a while: 
        self.rb.sim.step()

        if (self.reward() != 0.0) or (sum([self.punish(i) for i in range(len(self.add_state_num_list))]) != 0.0):
            self.rb.sim.set_state(self.init_state)
            # init qpos:
            init_qpos = self.choose_init_qpos()
            self.action(init_qpos)
            # clear the force on object:
            self.step_walk([0.0] * self.state_num)
            # clear the force on obstcle
            for i, l in enumerate(self.add_action_num_list):
                self.step_add_walk([0.0]*l, i)        # init velocity: we decrease the upper bound
        self.rb.sim.step() # step when qvel doesn't exist
        self.random_qvel(0.5 * random_scale)
        
        return self.get_state('state'), self.get_state('target'), self.get_state('add')

    def random_qvel(self, random_scale):
        # random_scale is 0~10
        # random qvel of -0.5 ~ 0.5
        for i in range(self.state_num):
            self.rb.sim.data.qvel[i] = random_scale * 0.2 * (random.random() - 0.5)

    def get_scaler(self):
        ''' get the scale and offset for each joint 
            mapping -1~1 -> 0.1*( range[0], range[1])
            force only exists in state and add
        '''
        ranges = self.rb.model.actuator_ctrlrange 
        self.scaler = []
        for r in ranges:
            scale = (r[1] - r[0])/2.0
            low_bound = r[0]
            up_bound = r[1]
            self.scaler.append([scale, low_bound, up_bound])
    
    def get_range_scaler(self):
        ''' get range scaler, the range scaler is of vital importance to 
        random initation.
        '''
        ranges = self.rb.model.jnt_range 
        self.range_scaler = []
        init_qpos = self.choose_init_qpos()
        for r, iq in zip(ranges, init_qpos):
            scale = (r[1] - r[0])/10.0
            low_bound = r[0]
            high_bound = r[1]
            self.range_scaler.append([low_bound, high_bound, iq, scale])

    def action(self, state):
        old_state = self.rb.get_qpos()
        a_num = np.array(state).shape[0]
        old_state[:a_num] = state
        self.rb.set_by_type(old_state)

    def vel_action(self, state):
        a_num = np.array(state).shape[0]
        for i, s in enumerate(state):
            self.rb.sim.data.qvel[i] = s

    # action:
    def step_walk(self, state):
        # clip the data
        state = np.clip(state, -1.0, 1.0)
        for i, s in enumerate(self.scaler[:self.state_num]):
            self.rb.sim.data.ctrl[i] = np.clip(state[i]*s[0], s[1], s[2]) 

    def step_add_walk(self, state, no):
        ''' run the add.no 
        '''
        low_ec = self.acc_add_action_num_list[no] + self.state_num
        high_ec = self.acc_add_action_num_list[no + 1] + self.state_num
        
        for i, s in enumerate(self.scaler[low_ec:high_ec]):
            self.rb.sim.data.ctrl[i + low_ec] = np.clip(state[i]*s[0], s[1], s[2]) 

    # obs:
    def get_state(self, cl):
        if cl == 'state':
            pos_state = self.rb.get_qpos()[:self.state_num]
            vel_state = self.rb.get_qvel()[:self.state_num]
            state = np.append(pos_state, vel_state)
        elif cl == 'target':
            pos_state = self.rb.get_qpos()[self.state_num: self.state_num + self.target_num]
            vel_state = self.rb.get_qvel()[self.state_num: self.state_num + self.target_num]
            state = np.append(pos_state, vel_state)
        else:
            # add state:
            state = []
            for i in range(len(self.add_state_num_list)):
                low_ec = self.acc_add_state_num_list[i] + self.state_num + self.target_num
                high_ec = self.acc_add_state_num_list[i + 1] + self.state_num + self.target_num
                pos_state = self.rb.get_qpos()[low_ec: high_ec]
                vel_state = self.rb.get_qvel()[low_ec: high_ec]
                state.append(np.append(pos_state, vel_state))
        return state

    def batch_add_states(self, add_states):
        if add_states == []:
            return []
        else:
            self.split_points = [0]
            for i in add_states:
                self.split_points.append(i.shape[0] + self.split_points[-1])
            return np.concatenate(add_states, axis = -1)

    def split_add_states(self, add_states):
        s_add_states = []
        for i in range(len(self.split_points) - 1):
            l_b = self.split_points[i]
            h_b = self.split_points[i+1]
            s_add_states.append(np.stack(add_states[:, l_b:h_b]))
            
        return s_add_states

    def reward(self):
        # when target moves, it means collision happens
        target_condition = sum(self.rb.get_qvel()[self.state_num: self.state_num + self.target_num])
        # eps = 1e-4
        if (self.rb.get_sensor()[0] != 0.0) or (target_condition != 0.0):
            return self.rp_list[0]
        else:
            return 0.0

    def punish(self, no):
        ''' Args:
            no, the no_th obstacle
        '''
        flag = False
        if self.attribute_list[no] == 'traffic':
            # punish for traffic limit:
            speed_limit = self.get_state('add_state')[no][0] # the position of speed label
            robot_speed = np.absolute(self.rb.get_qvel()[:self.state_num]).max() # max speed limit or average?
            over_speed = (robot_speed - speed_limit) if (robot_speed > speed_limit) else 0
            return self.rp_list[no + 1] * over_speed
            
        if self.add_state_num_list != 0:
            start = self.state_num + self.target_num
            pos = start + self.acc_add_state_num_list[no]
            flag = (self.rb.get_qvel()[pos] != 0)

        if (self.rb.get_sensor()[no + 1] != 0.0) or flag:
            return self.rp_list[no + 1]
        else:
            return 0.0

    # step: 
    def step(self, action):
        # different task share the same action
        # time control first:
        if 'door' in self.attribute_list:
            if self.time > self.step_num:
                self.default_add_action([1.0], self.door_num) # the door is usually the 0's action

        if 'traffic' in self.attribute_list:
            if self.time % self.traffic_interval == 0: # reverse at first time
                self.direction = - self.direction
            self.default_add_action([self.direction], self.traffic_num) # the door is usually the 0's action

        self.time += 1.0
        self.default_action(action)
        self.rb.sim.step()
        state = self.get_state('state')
        target_state = self.get_state('target')
        add_state_list = self.get_state('add_state')
        reward = self.reward()
        punish_list = [self.punish(i) for i in range(len(self.add_state_num_list))]
        done = reward  # when getting reward, it is done.
        # increase reward a lot 
        reward = reward + sum(punish_list)

        return state, target_state, add_state_list, reward, done, punish_list
    
    def inverse_set(self, state, target_state, add_state):
        # set qpos:    
        # remove time from state:
        state = state[:2*self.state_num]
        state_qpos = state[:self.state_num] 
        state_qvel = state[self.state_num:]
        target_qpos = target_state[:self.target_num]
        target_qvel = target_state[self.target_num:]
        add_qpos = []
        add_qvel = []
        for i, l in enumerate(self.add_state_num_list):
            add_qpos.append(add_state[i][:l])
            add_qvel.append(add_state[i][l:])
        add_qpos = np.array(add_qpos).reshape([-1])
        add_qvel = np.array(add_qvel).reshape([-1])
        # add all state:
        qpos = np.concatenate([state_qpos, target_qpos, add_qpos], axis = 0)
        qvel = np.concatenate([state_qvel, target_qvel, add_qvel], axis = 0)
        self.action(qpos) 
        self.vel_action(qvel)
        # forward () the whole state
        self.rb.sim.forward()
        return self.get_state('state'), self.get_state('target'), self.get_state('add')

    def set_start_pool(self, starts):
        # starts_pool should be a list
        self.start_pool = starts
        self.start_pool_num = len(starts)

    def reset_from_pool(self):
        num = random.randint(0, self.start_pool_num - 1)
        start = self.start_pool[num]
        self.inverse_set(start[0], start[1], start[2])
        return self.get_state('state'), self.get_state('target'), self.get_state('add')

    # image
    def get_picture(self):
        img =  self.rb.get_picture([image_size, image_size]) 
        return img

    def save_ex_img(self):
        img = self.get_picture()
        cv2.imwrite('test.jpg', img)

    def get_video(self, name, state_list, target_list, add_state_list):
        # get video from an state list:
        # record state is more precise
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video = cv2.VideoWriter('./{}.avi'.format(name), fourcc, 20, (image_size, image_size), True)
        for (state, target, add_state) in zip(state_list, target_list, add_state_list):
            self.inverse_set(state, target, add_state)
            picture = self.get_picture()
            video.write(picture)
        video.release()

    # utils
    def merge_state(self, state_list, target_state_list, add_state_list):
        ''' Change the state_list into pos '''
        num = np.array(state_list).shape[0]
        state = state_list[:, :self.state_num]
        if self.task_name != 'DA':
            target_state = target_state_list[:, :self.target_num]
        else:
            target_state = []
        add_state = []
        for i, l in enumerate(self.add_state_num_list):
            add_state.append(add_state_list[i][:, :l])

        if add_state != []:
            add_state = np.concatenate(add_state, -1)
        else:
            add_state = np.array([]).reshape([num,0])
        
        if self.task_name == 'DA':
            target_state = np.array([]).reshape([num,0])
        new_state = np.concatenate([state, target_state, add_state], axis = -1)
        return new_state


if __name__ == '__main__':
    # task test
    myenv = env('ball')
    myenv.step([1.0, 1.0])

    myenv = env('arm')
    myenv.step([1.0,]*5)

    myenv = env('3darm')
    myenv.step([1.0,]*5)

    # task test
    myenv = env('ball,safety')
    myenv.step([1.0, 1.0])

    myenv = env('arm,safety')
    myenv.step([1.0, ] * 5)

    myenv = env('ball,door')
    myenv.step([1.0, 1.0])

    myenv = env('arm,door')
    myenv.step([1.0, ] * 5)

    myenv = env('ball,safety,safety')
    myenv.step([1.0, 1.0])

    # exam for inverse settting:
    myenv = env('ball')
    myenv.step([1.0, 1.0])
    myenv.inverse_set([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [])
    print(myenv.get_state('state'))