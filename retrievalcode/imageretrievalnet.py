import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
# import pretrainedmodels
from pooling import MAC, SPoC, RMAC, RAMAC
import cv2
import random
from models import SENet154, ResNeXt101, load_state_dict
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import ImageData

class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class ImageRetrievalNet(nn.Module):

    def __init__(self, features, meta):
        super(ImageRetrievalNet, self).__init__()
        #self.features = nn.Sequential(*features)
        self.features = features
        self.norm = L2N()
        self.meta = meta

    def forward(self, x):
        # features -> pool -> norm
        x, feature_conv4 = self.features(x)
        # print(x.shape)
        # feature_MAC = self.norm(MAC()(x)).squeeze(-1).squeeze(-1)
        # feature_SPoC = self.norm(SPoC()(x)).squeeze(-1).squeeze(-1)
        feature_RMAC = self.norm(RMAC()(x)).squeeze(-1).squeeze(-1)
        # feature_RAMAC = self.norm(RAMAC()(x)).squeeze(-1).squeeze(-1)

        return feature_RMAC, feature_conv4


def init_network(model_choose, pretrain):
    MODEL={'senet154':SENet154, 'resnext101':ResNeXt101}[model_choose]
    if pretrain:
        print('loading pretrained model')
        net_in = MODEL(pretrained=True)
    else:
        model_file = '../traincode/experiments/metric/ResNeXt101_metric_5.pth'
        print('loading ' + model_file)
        net_in = MODEL(pretrained=False)
        load_state_dict(net_in, model_file)
    
    net_in.cuda()
    net_in.eval()    
    mean=net_in.mean
    std=net_in.std
    dim =  2048

    # create meta information to be stored in the network
    meta = {'architecture':model_choose, 'outputdim':dim, 'mean':mean, 'std':std}

    # create a generic image retrieval network
    net = ImageRetrievalNet(net_in.features, meta)

    return net

def gen_im_patch(img, imsize, split_num = 3, patch_num = 10):
    h, w, c = img.shape[:3]
    
    hs = h // split_num
    ws = w // split_num
    
    patches = np.empty((imsize, imsize, c, patch_num * split_num ** 2 + 1), dtype=img.dtype)
    for i in range(split_num):
        for j in range(split_num):
            if patch_num == 1:
                x = i*ws
                y = j*hs
                patch = img[y:y + hs, x:x + ws, :]
                patch = cv2.resize(patch, (imsize, imsize))
                patches[:, :, :, (i * split_num + j)] = patch
            else:
                for p in range(patch_num):
                    x = int(i * ws + ws * random.uniform(-0.25, 0.25))
                    y = int(j * hs + hs * random.uniform(-0.25, 0.25))
                    x = min(max(0, x), w - ws)
                    y = min(max(0, y), h - hs)
                    patch = img[y:y + hs, x:x + ws, :]
                    patch = cv2.resize(patch, (imsize, imsize))
                    patches[:, :, :, (i * split_num + j) * patch_num + p] = patch
    patches[:, :, :, -1] = cv2.resize(img, (imsize, imsize))
    return patches

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()
    
def extract_vectors(net, images, image_size):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    normalize = torchvision.transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    # extracting vectors
    #split_num = param.region_split
    #patch_num = param.patch_num
    batch_size=4
    loader = torch.utils.data.DataLoader(
        ImageData(images, transform, imsize = image_size), batch_size = batch_size, shuffle = False, num_workers = 12)    
    #N = patch_num * split_num ** 2 + 1
    #vecs_MAC = torch.zeros(len(images), net.meta['outputdim'])
    #vecs_SPoC = torch.zeros(len(images), net.meta['outputdim'])
    vecs_RMAC = torch.zeros(len(images), net.meta['outputdim'])
    #vecs_RAMAC = torch.zeros(len(images), net.meta['outputdim'])
    mats_conv4 = torch.zeros(len(images), 1024, 14, 14)
    name_list = []
    bar = tqdm(total=len(images))

    for i, data in enumerate(loader):
        bar.update(batch_size)
        inputs, names = data
        input_var = Variable(inputs.cuda())
        batch_range = range(i*batch_size, min(len(images), (i+1)*batch_size))
        feature_RMAC, feature_conv4 = net(input_var)
        #vecs_MAC[i] = feature_MAC.cpu().data.squeeze() # 2048
        #vecs_SPoC[i] = feature_SPoC.cpu().data.squeeze()
        vecs_RMAC[batch_range] = feature_RMAC.cpu().data.squeeze()
        #vecs_RAMAC[i] = feature_RAMAC.cpu().data.squeeze()
        mats_conv4[batch_range] = feature_conv4.cpu().data.squeeze()
        name_list.extend(names)


        #if (i+1) % print_freq == 0 or (i+1) == len(images):
        #    print('\r>>>> {}/{} done...'.format((i+1), len(images)))

    return vecs_RMAC, mats_conv4, name_list

