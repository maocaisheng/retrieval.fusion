import argparse
import os
import yaml

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from easydict import EasyDict
from tensorboardX import SummaryWriter

from tricks.training_refinements.LR import CosineAnnealing
from models import ResNeXt101
from dataset import PairData, makeloader, DataAug
# from utils import CenterCrop, PerClassAccuracy
from loss import MetricLoss

GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

parser = argparse.ArgumentParser(description='baseline training')
parser.add_argument('--config', default='experiments/metric/config.yaml')

def train(model, epoch_idx, contrastive, lr_scheduler, optimizer, trainloader):
    model.train()

    for batch_idx, data in enumerate(trainloader):
        images1, images2, labels = data
        images1 = images1.cuda()
        images2 = images2.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outs1 = model(images1)
        outs2 = model(images2)
        loss = contrastive(outs1, outs2, labels)
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        Writer.add_scalar('train loss', loss.item(), epoch_idx * len(trainloader) + batch_idx)

        print('TRAIN:{}/{} EPOCHs, {}/{} BATCHs, LOSS:{}'.format(epoch_idx, config.epoch_num, batch_idx,
                len(trainloader), loss.item()))

def val(model, epoch_idx, contrastive, testloader):
    model.eval()
    margin = 1.
    accuracy = []
    for batch_idx, data in enumerate(testloader):
        images1, images2, labels = data
        images1, images2, labels = images1.cuda(), images2.cuda(), labels.cuda()

        with torch.no_grad():
            # outs1 = model(images1).squeeze(-1).squeeze(-1)
            # outs2 = model(images2).squeeze(-1).squeeze(-1)
            # distance = (outs1-outs2).pow(2).sum(dim=1).sqrt()
            outs1 = model(images1)
            outs2 = model(images2)
            euclidean_distance = F.pairwise_distance(outs1, outs2, keepdim = False)
            pred = (euclidean_distance < margin) 
            # print(pred)
            acc = (pred == labels.to(torch.uint8)).float().mean()
            accuracy.append(acc.item())
            #loss = contrastive(outs1, outs2, labels)

        Writer.add_scalar('val acc', acc.item(), epoch_idx * len(testloader) + batch_idx)

        #print('VAL:{}/{} EPOCHs, {}/{} BATCHs, ACCURACY:{}'.format(epoch_idx, config.epoch_num, batch_idx, len(testloader), acc.item()))
    accuracy = sum(accuracy)/len(accuracy)
    print('>> VAL:{}/{} EPOCHs, MEAN ACCURACY:{}'.format(epoch_idx, config.epoch_num, accuracy))
    return accuracy

def save_checkpoints(model, acc, epoch_idx, suffix=''):
    infos = {
        "MODEL": model.state_dict(),
        "ACCURACY": acc,
        "EPOCH_IDX": epoch_idx,
    }
    torch.save(infos, os.path.join(config.save_path, config.model_type +'_'+ suffix +'_'+ str(epoch_idx) + '.pth'))
    
    
def main():
    global args, config
    args = parser.parse_args()
    with open(args.config) as rPtr:
        config = EasyDict(yaml.load(rPtr))

    config.save_path = os.path.dirname(args.config)
    print(config)

    # Random seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    train_trans = DataAug(config.image.resized, config.image.size, config.image.mean, config.image.std, istrain=True)
    val_trans = DataAug(config.image.resized, config.image.size, config.image.mean, config.image.std, istrain=False)


    trainsets = PairData(phase='train', transform=train_trans)
    print('>> number of train sets: {}'.format(len(trainsets)))
    trainloader = makeloader(trainsets, config.batch_size, shuffle=True, num_workers=config.num_workers)

    valsets = PairData(phase='val', transform=val_trans)
    valloader = makeloader(valsets, config.batch_size, shuffle=False, num_workers=config.num_workers)

    global Writer
    Writer = SummaryWriter(config.save_path + '/events')
    # Model
    model = ResNeXt101(pretrained=config.pretrained)
    # NOT support nested model
    # Writer.add_graph(model, (torch.zeros(3, 32, 32),))
    model = model.cuda()

    # Optimizer
    contrastive = MetricLoss.ContrastiveLoss(margin=1.)
#     frozen_params = list(map(id, model.layer0.parameters()))
#     backbone_params = filter(
#         lambda params: id(params) not in frozen_params,
#         model.parameters())
#     params = [
#         {'params': backbone_params, 'lr': config.lr_scheduler.base_cnn_lr},
#         {'params': model.layer0.parameters(), 'lr': 0}
#     ]
    params = [
        {'params': model.parameters(), 'lr': config.lr_scheduler.base_cnn_lr}
    ]
    optimizer = optim.SGD(params, momentum=config.momentum, weight_decay=config.weight_decay)

    # Learing rate scheduler
    lr_scheduler = CosineAnnealing(optimizer, len(trainloader) * config.epoch_num)

    BEST_ACC = 0
    for epoch_idx in range(config.epoch_num):
        train(model, epoch_idx, contrastive, lr_scheduler, optimizer, trainloader)
        acc = val(model, epoch_idx, contrastive, valloader)
        if BEST_ACC < acc:
            BEST_ACC = acc
            save_checkpoints(model, acc, epoch_idx, suffix='metric')
            print('>> save checkpoints, current epoch {}'.format(epoch_idx))

    Writer.close()

if __name__ == '__main__':
    main()
