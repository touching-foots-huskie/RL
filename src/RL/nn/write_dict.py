import collections


def write_dict(name='caln'):
    if name == 'caln':
        saver_dict = collections.OrderedDict()
        base_list = ['ball', 'arm']
        addition_list = ['safety', 'door', 'traffic', 'force', None]
        part_list = ['action', 'value', 'activation']
        # combination:
        for base in base_list:
            for addition in addition_list:
                for part in part_list:
                    if addition != None:
                        saver_dict['{}_{}_{}'.format(base, addition, part)] = 0
                    else:
                        saver_dict['{}_{}_{}'.format(base, base, part)] = 0

        for i, name in enumerate(saver_dict.keys()):
            saver_dict[name] = str(i)
        return saver_dict
    else:
        pass
