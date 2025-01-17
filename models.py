import torch
import torch.nn as nn
import torch.nn.functional as F
from transfer_losses import TransferLoss
import backbones
from lmmd_loss import LMMD_Loss
from mmd_loss import MMDLoss
class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='adv', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.lmmdloss = LMMD_Loss(**transfer_loss_args)
        self.mmdloss =MMDLoss(**transfer_loss_args)
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    # def forward(self, source, target, source_label):
    #     source = self.base_network(source)
    #     target = self.base_network(target)
    #     if self.use_bottleneck:
    #         source = self.bottleneck_layer(source)
    #         target = self.bottleneck_layer(target)
    #     # classification
    #     source_clf = self.classifier_layer(source)
    #     clf_loss = self.criterion(source_clf, source_label)#分类损失
    #     # transfer
    #     kwargs = {}
    #     if self.transfer_loss == "lmmd":
    #         kwargs['source_label'] = source_label
    #         target_clf = self.classifier_layer(target)
    #         kwargs['target_logits'] = F.softmax(target_clf, dim=1)
    #     elif self.transfer_loss == "daan":
    #         source_clf = self.classifier_layer(source)
    #         kwargs['source_logits'] = F.softmax(source_clf, dim=1)
    #         target_clf = self.classifier_layer(target)
    #         kwargs['target_logits'] = F.softmax(target_clf, dim=1)
    #     elif self.transfer_loss == 'bnm':
    #         tar_clf = self.classifier_layer(target)
    #         target = nn.Softmax(dim=1)(tar_clf)
    #
    #     transfer_loss = self.adapt_loss(source, target, **kwargs)
    #     return clf_loss, transfer_loss

    def forward(self, source, target, source_label):
        source = self.base_network(source)
        target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        # classification
        source_clf = self.classifier_layer(source)
        clf_loss1 = self.criterion(source_clf, source_label)#分类损失

        # transfer
        kwargs = {}
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        kwargs['source_label'] = source_label
        target_clf = self.classifier_layer(target)
        kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        lmmd_loss ,pseudo= self.lmmdloss(source, target, **kwargs)
        # mmd_loss = self.mmdloss(source,target)
        clf_loss = clf_loss1 + 0.1*lmmd_loss+0.01*pseudo #default clf_loss = clf_loss1 +
        # clf_loss = clf_loss1+0.1*mmd_loss
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},

            {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
        ]
        # if self.use_bottleneck:
        #     params.append(
        #         {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
        #     )
        #
        # #Loss-dependent
        # if self.transfer_loss == "adv":
        #     params.append(
        #         {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
        #     )
        # elif self.transfer_loss == "daan":
        #     params.append(
        #         {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
        #     )
        #     params.append(
        #         {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
        #     )
        return params
    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        return clf


    def extractor(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        # clf = self.classifier_layer(x)
        return x

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass