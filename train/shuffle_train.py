# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# shuffle train is used to store the decorator for shuffle train:
# $File basic_train.py
from sklearn.utils import shuffle


def train(shuffle_bool):
    def _train(func):
        def wrapper(*args, **kw):
            update_data, epochs, batch_num, update_func = func(*args, **kw)
            # get data_list, and name_list:

            if shuffle_bool:
                # shuffle
                data_list = []
                name_list = []
                for name, data in update_data.items():
                    name_list.append(name)
                    data_list.append(data)

                data_len = data_list[0].shape[0]
                # num_batches:
                num_batches = max(data_len // batch_num, 1)
                batch_size = data_len // num_batches
                data_list = shuffle(*data_list)
                for i in range(epochs):
                    for j in range(num_batches):
                        start = j * batch_size
                        end = (j + 1) * batch_size
                        update_data = {}
                        for name, data in zip(name_list, data_list):
                            update_data[name] = data[start:end, :]
                        update_func(update_data)
            else:
                for i in range(epochs):
                    update_func(update_data)

        return wrapper
    return _train