import mxnet as mx


def get_order_config(coop_configs):
    order_config = []
    for configs in coop_configs:
        for config in configs:
            if config not in order_config:
                order_config.append(config)

    return sorted(order_config)


def get_coop_config(coop_configs_str):
    coop_configs = []
    configs = list(filter(None, coop_configs_str.split(',')))
    if len(configs) != 3:
        raise Exception('coop configs should have three layers!')
    for config in configs:
        coop_configs.append(tuple(sorted(map(int, filter(None, config.split(' '))))))
    return tuple(coop_configs)


class LossMetric:

    def __init__(self, order_sig_config):
        self._order_sig_config = order_sig_config
        for sig_level in self._order_sig_config:
            self.__dict__['obj_sig{}'.format(sig_level)] = mx.metric.Loss('obj_sig{}'.format(sig_level))
            self.__dict__['xy_sig{}'.format(sig_level)] = mx.metric.Loss('xy_sig{}'.format(sig_level))
            self.__dict__['wh_sig{}'.format(sig_level)] = mx.metric.Loss('wh_sig{}'.format(sig_level))

        self.__dict__['cls'] = mx.metric.Loss('cls')

    def initial(self):
        for sig_level in self._order_sig_config:
            self.__dict__['obj_sig{}list'.format(sig_level)] = []
            self.__dict__['xy_sig{}list'.format(sig_level)] = []
            self.__dict__['wh_sig{}list'.format(sig_level)] = []
        self.__dict__['cls_list'] = []

    def append(self, loss_list):
        for sig_level in self._order_sig_config:
            self.__dict__['obj_sig{}list'.format(sig_level)].append(loss_list.pop(0))
            self.__dict__['xy_sig{}list'.format(sig_level)].append(loss_list.pop(0))
            self.__dict__['wh_sig{}list'.format(sig_level)].append(loss_list.pop(0))
        self.__dict__['cls_list'].append(loss_list.pop(0))

    def update(self):
        for sig_level in self._order_sig_config:
            self.__dict__['obj_sig{}'.format(sig_level)].update(0, self.__dict__['obj_sig{}list'.format(sig_level)])
            self.__dict__['xy_sig{}'.format(sig_level)].update(0, self.__dict__['xy_sig{}list'.format(sig_level)])
            self.__dict__['wh_sig{}'.format(sig_level)].update(0, self.__dict__['wh_sig{}list'.format(sig_level)])
        self.__dict__['cls'].update(0, self.__dict__['cls_list'])

    def get(self):
        name_loss = []
        name_loss_str = ''
        for sig_level in self._order_sig_config:
            name_loss_str += ', '
            name1, loss1 = self.__dict__['obj_sig{}'.format(sig_level)].get()
            name2, loss2 = self.__dict__['xy_sig{}'.format(sig_level)].get()
            name3, loss3 = self.__dict__['wh_sig{}'.format(sig_level)].get()
            name_loss += [name1, loss1, name2, loss2, name3, loss3]
            name_loss_str += '{}={:.3f}, {}={:.3f}, {}={:.3f}'
        name4, loss4 = self.__dict__['cls'].get()
        name_loss += [name4, loss4]
        name_loss_str += ', {}={:.3f}'
        return name_loss_str, name_loss
