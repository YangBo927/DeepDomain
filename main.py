import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
# from early_stopping import EarlyStopping
#
# early_stop_path = "./early_stop"
# if not os.path.exists(early_stop_path):
#     os.makedirs(early_stop_path)
#
# early_stopping = EarlyStopping(early_stop_path)

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # 这段代码是在使用configargparse库创建一个参数解析器（Argument Parser）对象parser，用于解析和处理配置文件中的参数。
    #
    # 具体来说，代码中的ArgumentParser类是configargparse库提供的一个类，用于创建命令行参数解析器。通过该解析器，我们可以定义和解析命令行参数，以及读取和解析配置文件中的参数。
    #
    # 在代码中，ArgumentParser类的构造函数接收一些参数来配置解析器的行为：
    #
    # description：描述解析器的简短说明。
    # config_file_parser_class：指定用于解析配置文件的解析器类，这里使用了configargparse.YAMLConfigFileParser，表示使用YAML格式的配置文件进行解析。
    # formatter_class：指定帮助信息的格式化类，这里使用了configargparse.ArgumentDefaultsHelpFormatter，表示在帮助信息中显示参数的默认值。

    # 通过创建parser对象后，我们可以进一步定义和添加各种命令行参数，包括位置参数、可选参数、配置文件参数等。然后，我们可以使用parser.parse_args()方法解析命令行参数，并获取参数的值进行后续处理。

    # general configuration
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=3)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    # parser.add_argument('--data_dir', type=str, required=True)
    # parser.add_argument('--src_domain', type=str, required=True)
    # parser.add_argument('--tgt_domain', type=str, required=True)
    
    # training related
    parser.add_argument('--batch_size', type=int, default=48)

    parser.add_argument('--n_epoch', type=int, default=40)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    # type = str2bool是一个自定义函数，用于将字符串转换为布尔值。它的作用是将字符串类型的"True"或"False"转换为对应的布尔值True或False。如果字符串不是"True" 或"False"，则会引发ValueError异常。
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-1)#default=0.01
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=False)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=1)#default=10
    parser.add_argument('--transfer_loss', type=str, default='adv')
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    data_dir ='/tmp/pycharm_project_681/DatasetNew'
    #data_dir = 'D:\Test_datasets\Datasets'
    src_domain = 'JUN1source'
    tgt_domain = 'JUN3target'
    folder_src = os.path.join(data_dir, src_domain)
    folder_tgt = os.path.join(data_dir, tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class
# 这个函数load_data(args)是用来加载源领域(source domain)和目标领域(target domain)的数据，详细解析如下：
#
# folder_src = os.path.join(args.data_dir, args.src_domain): 创建源领域数据的路径，args.data_dir是数据集的根目录，args.src_domain是源领域数据的子目录。
#
# folder_tgt = os.path.join(args.data_dir, args.tgt_domain): 创建目标领域数据的路径。
#
# source_loader, n_class = data_loader.load_data(...): 使用数据加载器加载源领域的数据。args.batch_size指定了每批数据的数量，infinite_data_loader=not args.epoch_based_training用于决定是否使用无限数据加载器，train=True表示这是训练模式。
#
# target_train_loader, _ = data_loader.load_data(...): 加载目标领域的训练数据，参数设置和加载源领域数据类似。第二个返回值使用了一个占位符"_"来忽略，因为类别数量n_class在加载源领域数据时已经获取。
#
# target_test_loader, _ = data_loader.load_data(...): 加载目标领域的测试数据，参数train=False表示这是测试模式，而infinite_data_loader=False表示测试过程中不需要无限数据加载器。
#
# 最后返回加载得到的源领域数据加载器source_loader，目标领域的训练数据加载器target_train_loader，目标领域的测试数据加载器target_test_loader，以及类别数量n_class。
#
# 这个函数中提到的无限数据加载器可能是一种机制，当数据过多时，无法一次性加载到内存中，或者在每个epoch中需要重新洗牌数据时（也就是args.epoch_based_training为True），就会需要这种无限数据加载器。
#
# 注意，load_data函数实际上在这段代码里并没有具体实现，但它应该是一个可以读取数据并返回PyTorch的DataLoader对象的函数。此外，还需注意处理数据的多线程加载，参数num_workers=args.num_workers决定了数据加载过程中使用的子进程数。

def get_model(args):
    model = models.TransferNet(
         args.n_class,transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # correct = torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    for e in range(1, args.n_epoch+1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        # 这段代码看起来是在定义一个名为train_loss_clf的对象，它是一个utils.AverageMeter()类型的平均计量器（AverageMeter）。
        # 平均计量器是一种用于计算平均值和方差的工具，通常用于机器学习中的训练过程中。在训练过程中，我们需要记录每个批次的损失值，并计算整个训练集的平均损失值，以便评估模型的性能。
        #
        # utils.AverageMeter() 是一个自定义的类，它包含了三个方法：
        # reset(): 重置计量器，清空所有记录的数值。
        # update(val, n=1): 记录一个新的数值val，并将其加入到计量器中。n参数表示记录的数值个数，默认为1。
        # avg: 返回计量器中所有数值的平均值。
        # 因此，在上述代码中，train_loss_clf对象被用作记录训练过程中分类器的损失值。每次更新时，我们可以使用update() 方法将新的损失值加入到计量器中。在训练完成后，可以使用avg方法获取整个训练集的平均损失值。

        model.epoch_based_processing(n_batch)
        # 这段代码看起来是在调用model对象的一个方法epoch_based_processing(n_batch)，该方法可能是用于处理每个epoch的函数。
        #
        # 具体来说，epoch_based_processing(n_batch)方法可能会接收一个参数n_batch，该参数表示当前epoch中的批次数。在训练过程中，我们通常需要在每个epoch结束时进行一些处理，例如记录损失值、保存模型等。
        # 这些处理通常是基于整个epoch的，因此我们需要在每个epoch结束时调用一个函数或方法来执行这些处理。
        # 根据方法名和参数，epoch_based_processing(n_batch)
        # 方法可能会在每个epoch结束时被调用，它可能会执行一些基于整个epoch的处理，例如计算整个epoch的平均损失值、更新学习率等。n_batch参数可能会用于确定当前epoch中的批次数，以便在处理过程中使用。具体实现可能会因模型而异，需要查看具体实现才能确定其功能。
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        # criterion = torch.nn.CrossEntropyLoss()
        for _ in range(n_batch):
            data_source, label_source = next(iter_source) # .next()
            data_target, _ = next(iter_target) # .next()
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)
            
            clf_loss, transfer_loss = model(data_source, data_target, label_source)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss#default loss = clf_loss + args.transfer_loss_weight * transfer_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    #318修改
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())

        # train_loss_clf.update(clf_loss.item())的意义是更新分类器（classifier）的训练损失（loss）。在机器学习中，损失函数（lossfunction）用于衡量模型预测结果与实际标签之间的差异。通过最小化损失函数，可以优化模型的性能。
        #
        # 在这里，clf_loss是分类器的训练损失值，通过调用item()方法将其转化为Python中的标量值。然后，train_loss_clf.update(clf_loss.item())将该损失值添加到训练损失的累计统计中，以便跟踪模型在训练过程中的性能。这通常是为了监控模型的训练进展，并在需要时进行调整和优化。
        #
        # 具体来说，train_loss_clf.update()可能是一个用于更新训练损失统计的函数或方法。通过多次调用该函数并传入每个训练步骤的损失值，可以计算出整个训练过程中的平均损失或其他统计指标。这有助于评估模型的训练效果，并可以在训练过程中进行动态调整和监控。

        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Val
        stop += 1
        test_acc, test_loss = test(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)

        # early_stopping(test_acc / len(target_test_loader), model)
        # # 达到早停止条件时, early_stop会被置为True
        # if early_stopping.early_stop:
        #     print("======Early stopping!=======")
        #     break  # 跳出迭代，结束训练
        #
    print('Transfer result: {:.4f}'.format(best_acc))

def main():

    parser = get_parser() #初始化一个解析器，它通常被用于解析从命令行输入的参数。在机器学习的训练过程中，这些参数可能包括学习率、批次大小、训练轮数等。
    args = parser.parse_args()#解析通过命令行输入的参数。
    # setattr(args, "--data_dir", "D:/Transfer pretreatment/dataset")
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))#添加一个新的属性args，这个属性用于指定模型运行的设备，是GPU（如果可用）还是CPU。
    print(args)
    set_random_seed(args.seed)#设置随机种子以保证结果的可复现性。
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)#加载数据。source_loader是源数据集loader，target_train_loader是目标数据集的训练loader，
                                                                                    #            target_test_loader是目标数据集的测试loader，n_class是分类任务的类别数量。
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)#初始化模型，这个模型应该是一个深度神经网络模型，用于迁移学习
    optimizer = get_optimizer(model, args)#初始化优化器，优化器用于更新模型参数。
    
    if args.lr_scheduler: #判断是否使用学习率调度器，如果参数args.lr_scheduler为真，那么就通过get_scheduler(optimizer, args)方法获取学习率调度器；否则，学习率调度器为None。学习率调度器可以在训练过程中动态地改变学习率。
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    # torch.save(model, "./weights/net.pth")  # entire network
    # torch.save(model.state_dict(), "./weights/netJ132.pth")

if __name__ == "__main__":
    main()
