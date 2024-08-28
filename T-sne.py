import data_loader
import torch
import main
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt_sne
from sklearn import datasets
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

#Resnet50 = models.resnet50(pretrained = True)

def plot_tsne(features, labels):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    import seaborn as sns

    class_num = 8#len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    print('calss_num:', class_num)
    latent = features
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)
    # plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  # 将对降维的特征进行可视化
    # plt.show()
    x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
    tsne_features = (tsne_features - x_min) / (x_max - x_min)

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

     #hex = ["#c957db", "#dd5f57","#57db30", "#b9db57"]  # 绿、红

    data_label = []
    for v in df.y.tolist():
        if v == 0:
            data_label.append("SBall_images")
        elif v == 1:
            data_label.append("SInner_images")
        elif v == 2:
            data_label.append("SNormal_images")
        elif v == 3:
            data_label.append("SOuter_images")
        elif v == 4:
            data_label.append("TBall_images")
        elif v == 5:
            data_label.append("TInner_images")
        elif v == 6:
            data_label.append("TNormal_images")
        elif v == 7:
            data_label.append("TOuter_images")

    df["value"] = data_label



    sns.scatterplot(x="comp-1", y="comp-2",hue=df.value.tolist(),
                    style=df.value.tolist(),
                    #palette=sns.color_palette("hls", len(np.unique(labels))),
                    palette = {"SBall_images":"#c957db","SInner_images": "#dd5f57", "SNormal_images":"#57db30","SOuter_images": "#b9db57", "TBall_images":"#e38c7a","TInner_images":"lime", "TNormal_images":"violet", "TOuter_images":"blue"},
                    markers={"SBall_images": "s", "SInner_images": "o", "SNormal_images": "h", "SOuter_images": "d","TBall_images": "s", "TInner_images": "o", "TNormal_images": "h", "TOuter_images": "d"},
                    s = 80,
                    data=df).set(title="")

    # sns.scatterplot(x="comp-1", y="comp-2",hue=df.y.tolist(),style=df.y.tolist(),
    #
    #                 palette=sns.color_palette("hls", class_num),
    #                 data=df).set(title="Bearing data T-SNE projection unsupervised")


    plt_sne.legend(loc="lower right",bbox_to_anchor=(1.47, 1.02))
    # 不要坐标轴
    plt_sne.axis("off")
    # plt_sne.savefig(os.path.join(fileNameDir, "%s.jpg") % str(epoch), format="jpg", dpi=300)
    plt_sne.show()

# 提取ResNet的输出特征
if __name__ == "__main__":
    place = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(place)
    parser = main.get_parser()
    args = parser.parse_args()
    dataloader1, n_class1 = data_loader.load_data('D:\Test_datasets\Datasets\CWRU1target', args.batch_size,
                                                infinite_data_loader=False, train=False,
                                                num_workers=args.num_workers)
    dataloader2, n_class2 = data_loader.load_data('D:\Test_datasets\Datasets\CWRU2target', args.batch_size,
                                                 infinite_data_loader=False, train=False,
                                                 num_workers=args.num_workers)
    setattr(args, "n_class",4)
    setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    setattr(args, "device", device)
    model = main.get_model(args)
    model.load_state_dict(torch.load("./weights/netC121.pth"))
    model.to(device)
    # model = Resnet50
    # model.to(device)

    features_combined = []
    targets_combined = []

    with torch.no_grad():
        for data, target in dataloader1:
            data, target = data.to(args.device), target.to(args.device)
            output = model.extractor(data)
            #output = model.forward(data)
            features_combined.append(output.cpu().numpy())  # 将张量转换为NumPy数组
            targets_combined.append(target.cpu().numpy())

        for data, target in dataloader2:
            data, target = data.to(args.device), target.to(args.device)
            target = target+4
            output = model.extractor(data)
            #output = model.forward(data)
            features_combined.append(output.cpu().numpy())  # 将张量转换为NumPy数组
            targets_combined.append(target.cpu().numpy())

    features_combined = np.concatenate(features_combined)
    targets_combined = np.concatenate(targets_combined)


    print(features_combined.shape)
    print(targets_combined.shape)
    plot_tsne( features_combined, targets_combined)
