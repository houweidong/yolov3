import logging
# from logging import handlers


class Logger(object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info'):
        # fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        # filename there just a name
        self.logger = logging.getLogger(filename)
        # format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        # sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        fh = logging.FileHandler(filename)
        # fh.setFormatter(format_str)
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(fh)

    @staticmethod
    def get_logger(filename):
        return logging.getLogger(filename)
# if __name__ == '__main__':
#     log = Logger('ap_results.log', level='debug')
#     log.logger.debug('debug')
#     log.logger.info('info')
