# $File robot.py
# $Author Harvey Chang<chnme40cs@gmail.com>

# this file defines the behavior of robot
# the new robot has several qpos

import mujoco_py as mj
import cv2
import numpy as np

class robot:
    
    def __init__(self, filename, n_step=1):
        self.model = mj.load_model_from_path(filename)
        self.sim = mj.MjSim(self.model, nsubsteps = n_step)

    def get_qpos(self):
        return self.sim.get_state().qpos

    def get_qvel(self):
        return self.sim.get_state().qvel

    def get_time(self):
        return self.sim.get_state().time

    def get_sensor(self):
        return self.sim.data.sensordata    
    
    def get_picture(self, size, camera = 0):
        camera_name = "camera" + str(camera + 1)
        img = self.sim.render(width = size[0], height = size[1], camera_name = camera_name)
        return img
    
    # set functions:
    def set_by_type(self, value, data_type = "qpos"):
        data = self.sim.get_state()
        # use another data = transfer
        data = np.array(data)
        # assign data
        if data_type == "qpos":        
            data[1] = value
        elif data_type == "time":
            data[0] = value
        elif data_type == "qvel":
            data[2] = value
        elif data_type == "action":
            data[3] = value
        else:
            pass
        # transfer back to 5 values:
        time, qpos, qvel, act, udd_state = data
        self.sim.set_state(mj.MjSimState(
        time, qpos, qvel, act, udd_state
        ))
        self.sim.forward()
        
    def run(self):
        self.sim.step() 
      
