# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# this file is used to give illustrative figures:
import numpy as np
from matplotlib import pyplot as plt


def result2np(results, name_list, source_dir='./figure'):
    # this function is used to transfer the result structure into np
    # name is the name base for name_list:
    for item in name_list:
        np.save('{}/{}'.format(source_dir, item), results[item])


def load_draw(name, source_dir='./figure'):
    y = np.load('{}/{}'.format(source_dir, name))
    # y is [N, ?]
    if y.shape[1] == 1:
        x = np.array(range(y.shape[0]))
    elif y.shape[1] == 2:
        x = y[:, 0]
        y = y[:, 1]
    else:
        print('such shape have not designed')

    fig = plt.plot(x, y)
    plt.show(fig)


if __name__ == '__main__':
    pass
