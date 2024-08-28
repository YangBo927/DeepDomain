import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    # 这段代码是一个定义了MMDLoss类的Python类。在类的初始化方法__init__中，接受了一些参数并将它们存储为类的属性。
    #
    # kernel_type = 'rbf': 这是一个参数的默认值，表示核函数的类型，默认是高斯径向基函数（RBF）。
    #
    # kernel_mul = 2.0: 这是另一个参数的默认值，用于指定核函数的一个参数。
    #
    # kernel_num = 5: 这个参数用于指示使用多少个核函数。
    #
    # fix_sigma = None: 这个参数是可选的，用于指定固定的高斯核函数的标准差。
    #
    # ** kwargs: 这个参数表示接受任意数量的关键字参数，并将其存储为字典形式的属性。
    #
    # 在初始化方法中，这些参数被保存为类的属性，以便在类的其他方法中使用。例如，在后续的方法中根据self.kernel_type的取值来执行不同的计算逻辑。

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    # 这段代码定义了一个gaussian_kernel方法，用于计算高斯核函数的值。让我来解释一下这段代码的逻辑：
    #
    # 首先，计算总样本数n_samples，并将源数据和目标数据连接在一起，形成total。
    #
    # 然后，使用torch.cat方法将源数据和目标数据连接在一起，并使用unsqueeze和expand方法构造 total0和total1，以便后续计算。
    #
    # 接下来计算total0和total1间的L2距离，即欧氏距离的平方，得到L2_distance。
    #
    # 根据是否给定固定的高斯核函数标准差，确定带宽bandwidth的值。如果给定了fix_sigma，则使用其值作为带宽；否则，计算样本间距离的均值作为带宽。
    #
    # 根据带宽、kernel_mul和kernel_num计算出一系列不同带宽的高斯核函数。
    #
    # 计算不同带宽下的高斯核函数值，并返回这些值的总和作为最终的核函数值。
    #
    # 这段代码实现了高斯核函数的计算过程，根据输入的参数计算出不同带宽下的高斯核函数值，并将它们进行加和，得到最终的核函数值。

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
