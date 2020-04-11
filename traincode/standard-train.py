import argparse
import os
import yaml

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from easydict import EasyDict
from tensorboardX import SummaryWriter

from tricks.training_refinements.LR import CosineAnnealing
# from model import resnet
from models import SENet154
from dataset import FGV6, makeloader
from utils import CenterCrop, PerClassAccuracy

GPU_ID = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


parser = argparse.ArgumentParser(description='fgcv6 of baseline training')
parser.add_argument('--config', default='experiments/standard/config.yaml')


def train(model, epoch_idx, criterion, lr_scheduler, optimizer, trainloader):
    model.train()

    for batch_idx, data in enumerate(trainloader):
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        out = model(images)
        pred = out.data.max(1)[1]
        PCA.update(labels.data.cpu().numpy(), pred.data.cpu().numpy())
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        Writer.add_scalar('train loss', loss.item(), epoch_idx * len(trainloader) + batch_idx)

        print('TRAIN:{}/{} EPOCHs, {}/{} BATCHs, LOSS:{}'.format(epoch_idx, config.max_iter, batch_idx,
                len(trainloader), loss.item()))

    _, _, mAP = PCA.calc()
    Writer.add_scalar('train mAP', mAP, epoch_idx)
    PCA.reset()

def val(model, epoch_idx, criterion, testloader):
    model.eval()

    for batch_idx, data in enumerate(testloader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            out = model(images)
            pred = out.data.max(1)[1]
            PCA.update(labels.data.cpu().numpy(), pred.data.cpu().numpy())
            loss = criterion(out, labels)

        Writer.add_scalar('val loss', loss.item(), epoch_idx * len(testloader) + batch_idx)

        print('VAL:{}/{} EPOCHs, {}/{} BATCHs'.format(epoch_idx, config.max_iter, batch_idx, len(testloader)))

    per_class_acc, AVG_acc, mAP = PCA.calc()
    Writer.add_scalar('avg acc', AVG_acc, epoch_idx)
    Writer.add_scalar('val mAP', mAP, epoch_idx)
    for class_idx in range(config.num_classes):
        Writer.add_scalar('val_class:' + str(class_idx) + '_acc', per_class_acc[class_idx], epoch_idx)
    PCA.reset()

    return AVG_acc, mAP

def save_checkpoints(model, accuracy, mAP, epoch_idx, surffix=''):
    infos = {
        "MODEL": model.state_dict(),
        "ACCURACY": accuracy,
        "mAP": mAP,
        "EPOCH_IDX": epoch_idx,
    }
    torch.save(infos, os.path.join(config.save_path, config.model_type + surffix +'new'+ '.pth'))


def main():
    global args, config
    args = parser.parse_args()
    with open(args.config) as rPtr:
        config = EasyDict(yaml.load(rPtr))

    config.save_path = os.path.dirname(args.config)

    # Random seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    train_trans = transforms.Compose([
        transforms.RandomResizedCrop((config.image.size, config.image.size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize(config.image.mean, config.image.std)
    ])

    val_trans = transforms.Compose([
        CenterCrop(config.image.size, config.image.resized, interpolation=config.image.interpolation),
        transforms.ToTensor(),
        transforms.Normalize(config.image.mean, config.image.std)
    ])

    trainsets = FGV6(phase='train', transform=train_trans)
    print(len(trainsets))
    trainloader = makeloader(trainsets, config.batch_size, shuffle=True, num_workers=config.num_workers)

    valsets = FGV6(phase='val', transform=val_trans)
    valloader = makeloader(valsets, config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Model
    print(config.pretrained, type(config.pretrained))
    print(config.lr_scheduler.base_cnn_lr, type(config.lr_scheduler.base_cnn_lr))
    print(config.lr_scheduler.base_lt_lr, type(config.lr_scheduler.base_lt_lr))
    model = SENet154(num_classes=config.num_classes, pretrained=config.pretrained)
    model = model.cuda()

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    lt_params = list(map(id, model.last_linear.parameters()))
    backbone_params = filter(
        lambda params: id(params) not in lt_params,
        model.parameters())
    params = [
        {'params': backbone_params, 'lr': config.lr_scheduler.base_cnn_lr},
        {'params': model.last_linear.parameters(), 'lr': config.lr_scheduler.base_lt_lr}
    ]
    optimizer = optim.SGD(params, momentum=config.momentum, weight_decay=config.weight_decay)

    # Learing rate scheduler
    lr_scheduler = CosineAnnealing(optimizer, len(trainloader) * config.max_iter)

    global PCA, Writer
    PCA = PerClassAccuracy(num_classes=config.num_classes)
    Writer = SummaryWriter(config.save_path + '/events')
    BEST_mAP = 0.0
    for iter_idx in range(config.max_iter):
        train(model, iter_idx, criterion, lr_scheduler, optimizer, trainloader)
        AVG_acc, mAP = val(model, iter_idx, criterion, valloader)
        if mAP >= BEST_mAP:
            BEST_mAP = mAP
            save_checkpoints(model, AVG_acc, BEST_mAP, iter_idx)

    Writer.close()

if __name__ == '__main__':
    main()
