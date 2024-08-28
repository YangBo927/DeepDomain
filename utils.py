class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


# 这段代码包含了一个名为 AverageMeter 的类和一个名为 str2bool 的函数。
#
# AverageMeter 类用于计算并存储数值的平均值和当前值。它具有以下几个方法：
#
# __init__(self): 类的初始化方法，调用 reset() 方法进行重置操作。
#
# reset(self): 重置所有统计量，包括 val（当前值）、avg（平均值）、sum（总和）和 count（数量）。
#
# update(self, val, n=1): 更新统计量，传入当前值 val 和样本数量 n。它会根据新的数值更新 sum 和 count，然后重新计算 avg。
#
# str2bool 函数用于将输入的字符串转换为布尔值。具体来说，如果输入是布尔型变量，则直接返回该变量；如果输入是字符串，则根据字符串的取值进行转换。例如，'yes', 'true', 't', 'y', '1' 对应 True，'no', 'false', 'f', 'n', '0' 对应 False。如果输入的字符串不在这些取值范围内，则会引发 ValueError 异常。
#
# 这些代码看起来像是用于训练过程中的统计和参数解析。AverageMeter 可能被用来跟踪损失值的平均和当前值，而 str2bool 则可能用于解析命令行参数中的布尔类型参数。