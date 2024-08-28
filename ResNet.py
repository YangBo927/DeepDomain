import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm
import backbones
import data_loader
import main


def get_model():
    model = backbones.get_backbone('resnet50').to(args.device)
    return model

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    data_dir ='/tmp/pycharm_project_681/Datasets'
    #data_dir = 'D:\Test_datasets\Datasets'
    src_domain = 'CWRU1source'
    tgt_domain = 'CWRU3target'
    folder_src = os.path.join(data_dir, src_domain)
    folder_tgt = os.path.join(data_dir, tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_test_loader, n_class

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer



def train_model(model,device,train_loader,val_loader,optimizer,criterion):
    model.train()
    total_loss = 0.0
    val_loss = 0.0
    val_acc = 0.0
    for batch_id, (images, labels) in enumerate(train_loader):
        # 部署到device上
        images, labels = images.to(device), labels.to(device)
        # 梯度置0
        optimizer.zero_grad()
        # 模型输出
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失
        total_loss += loss.item() * images.size(0)
        # 平均训练损失
    train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 前向传播输出
            loss = criterion(outputs, labels)  # 损失
            val_loss += loss.item() * images.size(0)  # 累计损失
            _, pred = torch.max(outputs, dim=1)  # 获取最大概率的索引
            correct = pred.eq(labels.view_as(pred))  # 返回：tensor([ True,False,True,...,False])
            accuracy = torch.mean(correct.type(torch.FloatTensor))  # 准确率
            val_acc += accuracy.item() * images.size(0)  # 累计准确率
        # 平均验证损失
        val_loss = val_loss / len(val_loader.dataset)
        # 平均准确率
        val_acc = val_acc / len(val_loader.dataset)
    return train_loss, val_loss, val_acc

def train_epochs(model,DEVICE,dataloaders, criterion, optimizer, epochs):
    # 输出信息
    print("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}".format('Epoch', 'Train Loss', 'val_loss', 'val_acc', 'Test Loss', 'Test_acc'))
    # 初始最小的损失
    # 开始训练、测试
    for epoch in range(epochs):
        # 训练，return: loss
        train_loss, val_loss, val_acc = train_model(model, DEVICE, dataloaders['train'], dataloaders['val'], optimizer, criterion)



if __name__ == '__main__':
    # 3 定义超参数

    parser = main.get_parser()  # 初始化一个解析器，它通常被用于解析从命令行输入的参数。在机器学习的训练过程中，这些参数可能包括学习率、批次大小、训练轮数等。
    args = parser.parse_args()
    setattr(args, "device", torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))  # 添加一个新的属性args，这个属性用于指定模型运行的设备，是GPU（如果可用）还是CPU。
    print(args)
    model = get_model()
    optimizer = get_optimizer(model, args)
    criterion = nn.NLLLoss()
    train_epochs(model,args.device,load_data(args), criterion, optimizer, epochs=40)