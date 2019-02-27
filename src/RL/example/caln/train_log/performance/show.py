import numpy as np
import pdb
from matplotlib import pyplot as plt

name = ["ball,safety_reward.npy",  "ball,safety_random.npy"]
reward = np.load(name[0])
random = np.load(name[1])
# pdb.set_trace()
plt.figure()
plt.plot(np.array(range(reward.shape[0])), reward)
plt.figure()
plt.plot(np.array(range(random.shape[0])), random)
plt.show()

