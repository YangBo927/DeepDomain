from torchvision import models
import torch
import torch.nn as nn
############################### CBAM程序  ###############################################
import torch.nn.functional as F
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(ChannelAttentionModule, self).__init__()
#         # 使用自适应池化缩减map的大小，保持通道不变
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x))
#         maxout = self.shared_MLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)
#
#
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # map尺寸不变，缩减通道
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out
#
#
# class CBAM_custom(nn.Module):
#     def __init__(self, channel):
#         super(CBAM_custom, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()
#
#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         return out
#
#
# class InceptionA(nn.Module):
#     def __init__(self, in_channels):
#         super(InceptionA, self).__init__()
#         # 第二个分支
#         self.branch1_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
#         # 第三个分支
#         self.branch5_5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
#         self.branch5_5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
#         # 第四个分支
#         self.branch3_3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
#         self.branch3_3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
#         self.branch3_3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
#         # 第一个分支
#         self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
#
#     def forward(self, x):
#         #         x = self.branch1×1(x)
#
#         #         x = self.branch5×5_1(x)
#         #         x = self.branch5×5_2(x)
#         # 以上两种写法是错误的，因为等式左边都是 x，而根据图片可知，各分支之间是并联结构。
#         branch1_1 = self.branch1_1(x)
#
#         branch5_5 = self.branch5_5_1(x)
#         branch5_5 = self.branch5_5_2(branch5_5)
#
#         branch3_3 = self.branch3_3_1(x)
#         branch3_3 = self.branch3_3_2(branch3_3)
#         branch3_3 = self.branch3_3_3(branch3_3)
#
#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)
#
#         outputs = [branch1_1, branch5_5, branch3_3, branch_pool]
#         return torch.cat(outputs, dim=1)  # (b, c, w, h),则dim=1，即按照通道进行拼接。
###############################################################################################


import torch
from torch import nn

# def last_zero_init(m):
#     if isinstance(m, nn.Sequential):
#         nn.init.constant_(m[-1].weight, 0)
#         m[-1].inited = True
#     else:
#         nn.init.constant_(m.weight, 0)
#         m.inited = True
#
# class ContextBlock2d(nn.Module):
#
#     def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=8):
#         super(ContextBlock2d, self).__init__()
#         assert pool in ['avg', 'att']
#         assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
#         assert len(fusions) > 0, 'at least one fusion should be used'
#         self.inplanes = inplanes
#         self.planes = planes
#         self.pool = pool
#         self.fusions = fusions
#         if 'att' in pool:
#             self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
#             self.softmax = nn.Softmax(dim=2)
#         else:
#             self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         if 'channel_add' in fusions:
#             self.channel_add_conv = nn.Sequential(
#                 nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
#                 nn.LayerNorm([self.planes // ratio, 1, 1]),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
#             )
#         else:
#             self.channel_add_conv = None
#         if 'channel_mul' in fusions:
#             self.channel_mul_conv = nn.Sequential(
#                 nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
#                 nn.LayerNorm([self.planes // ratio, 1, 1]),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
#             )
#         else:
#             self.channel_mul_conv = None
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.pool == 'att':
#             nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in')
#             self.conv_mask.inited = True
#
#         if self.channel_add_conv is not None:
#             last_zero_init(self.channel_add_conv)
#         if self.channel_mul_conv is not None:
#             last_zero_init(self.channel_mul_conv)
#
#     def spatial_pool(self, x):
#         batch, channel, height, width = x.size()
#         if self.pool == 'att':
#             input_x = x
#             input_x = input_x.view(batch, channel, height * width)
#             input_x = input_x.unsqueeze(1)
#             context_mask = self.conv_mask(x)
#             context_mask = context_mask.view(batch, 1, height * width)
#             context_mask = self.softmax(context_mask)
#             context_mask = context_mask.unsqueeze(3)
#             context = torch.matmul(input_x, context_mask)
#             context = context.view(batch, channel, 1, 1)
#         else:
#             context = self.avg_pool(x)
#
#         return context
#
#     def forward(self, x):
#         context = self.spatial_pool(x)
#
#         if self.channel_mul_conv is not None:
#             channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
#             out = x * channel_mul_term
#         else:
#             out = x
#         if self.channel_add_conv is not None:
#             channel_add_term = self.channel_add_conv(context)
#             out = out + channel_add_term
#
#         return out

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    # elif "alexnet" == name.lower():
    #     return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()

class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim
    
# convnet without the last layer
# class AlexNetBackbone(nn.Module):
#     def __init__(self):
#         super(AlexNetBackbone, self).__init__()
#         model_alexnet = models.alexnet(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
#         self.features = model_alexnet.features
#         self.classifier = nn.Sequential()
#         for i in range(6):
#             self.classifier.add_module(
#                 "classifier"+str(i), model_alexnet.classifier[i])
#         self._feature_dim = model_alexnet.classifier[6].in_features
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256*6*6)
#         x = self.classifier(x)
#         return x
#
#     def output_num(self):
#         return self._feature_dim

# class ResNetBackbone(nn.Module):
#     def __init__(self, network_type):
#         super(ResNetBackbone, self).__init__()
#         resnet = resnet_dict[network_type](weights=torchvision.models.resnet50().IMAGENET1K_V1)
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4
#         self.avgpool = resnet.avgpool
#         self._feature_dim = resnet.fc.in_features
#         del resnet
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         return x
    import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        # self.layer1 = nn.Sequential(resnet.layer1[0], CBAM_custom(256), resnet.layer1[1], CBAM_custom(256), resnet.layer1[2], CBAM_custom(256))
        # self.layer2 = nn.Sequential(resnet.layer2[0], CBAM_custom(512), resnet.layer2[1], CBAM_custom(512), resnet.layer2[2], CBAM_custom(512), resnet.layer2[3], CBAM_custom(512))
        # self.layer3 = nn.Sequential(resnet.layer3[0], CBAM_custom(1024), resnet.layer3[1], CBAM_custom(1024), resnet.layer3[2], CBAM_custom(1024), resnet.layer3[3], CBAM_custom(1024), resnet.layer3[4], CBAM_custom(1024))
        #self.layer4 = nn.Sequential(resnet.layer4[0], CBAM_custom(2048), resnet.layer4[1], CBAM_custom(2048), resnet.layer4[2], CBAM_custom(2048))
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        #self.layer3 = resnet.layer3
        self.layer3 = nn.Sequential(resnet.layer3[0], resnet.layer3[1], resnet.layer3[2], resnet.layer3[3] , resnet.layer3[4],)
        #self.layer4 = resnet.layer4
        #self.layer4 = nn.Sequential(resnet.layer4[0], resnet.layer4[1], resnet.layer4[2])
        self.layer4 = nn.Sequential(resnet.layer4[0], resnet.layer4[1], resnet.layer4[2])##,ContextBlock2d(2048,2048)
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim



