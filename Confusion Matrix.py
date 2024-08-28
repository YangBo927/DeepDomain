import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import data_loader
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import torch
from tqdm import tqdm
import main

import json

# 计算混淆矩阵
class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):  # 计算混淆矩阵的值
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):  # 计算各项指标
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 对角线的和
        acc = sum_TP / np.sum(self.matrix)  # 混淆矩阵的和
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]  # 表格的tittle
        for i in range(self.num_classes):
            TP = self.matrix[i, i]  # label为真，预测为真
            FP = np.sum(self.matrix[i, :]) - TP  # label为假，预测为真
            FN = np.sum(self.matrix[:, i]) - TP  # label为假，预测为真
            TN = np.sum(self.matrix) - TP - FP - FN  # label为假，预测为假
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)  # 设置x轴坐标label
        plt.yticks(range(self.num_classes), self.labels)  # 设置y轴坐标label
        plt.colorbar()  # 显示 colorbar

        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2  # 在图中标注数量/概率信息
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()




if __name__ == '__main__':
    place = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(place)
    parser = main.get_parser()
    args = parser.parse_args()
    dataloader, n_class = data_loader.load_data('D:\Test_datasets\Datasets\CWRU1target', args.batch_size,
                                                  infinite_data_loader=False, train=False,
                                                  num_workers=args.num_workers)

    setattr(args, "n_class", n_class)
    setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    setattr(args, "device", device)
    model = main.get_model(args)
    model.load_state_dict(torch.load("./weights/netC132.pth"))
    model.to(device)
    # model = Resnet50
    # model.to(device)

    # 类别
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # labels = [label for label in classes]
    # confusion = ConfusionMatrix(num_classes=10, labels=labels)

    # 类别
    json_label_path = './class_indices.json'
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=4, labels=labels)

    model.eval()
    with torch.no_grad():
        for data,target in dataloader:
            data = data.to(args.device)
            outputs = model.predict(data.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), target.to("cpu").numpy())  # 更新混淆矩阵的值
    confusion.plot()  # 绘制混淆矩阵
    confusion.summary()  # 计算指标
