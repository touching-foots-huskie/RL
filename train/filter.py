# Author: Harvey Chang
# Email: chnme40cs@gmail.com
# Filter is used to filter more valuable specimen from the results:


def filter(results, threshold, ratio):
    '''
    filter is used to filter bad performance.
    :param results: results: results is in the version of result list:
    :param threshold: the threshold to surpass
    :param ratio: ratio of choose
    :return: filtered results:
    '''
    whole_length = len(results['rewards'])
    filtered_results = []
    gate = whole_length * ratio  # for example: ratio is 0.5, whole_len is 40, then gate is 20
    for num, result in enumerate(results):
        if num < gate:
            if result['eval_value'] < threshold:
                filtered_results.append(result)  # collect bad examples
        else:
            filtered_results.append(result)
    return filtered_results   # can it be dealt with inside tensorflow?