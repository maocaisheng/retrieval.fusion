import os
import pickle
from PIL import Image
import torch
import torch.utils.data as data

def imresize(img,imsize):
    #img.thumbnail((imsize,imsize),Image.ANTIALIAS)
    img=img.resize((imsize, imsize),Image.ANTIALIAS)
    return img

class ImageData(data.Dataset):
    def __init__(self, img_list, transform = None,imsize = 480):
        self.img_list = img_list
        self.transform = transform
        self.imsize = imsize
    
    def __getitem__(self,index):
        
        img = self.img_list[index]
        im = Image.open(img).convert('RGB')
        im = imresize(im,self.imsize)
        
        if self.transform is not None:
            im = self.transform(im)
            
        return im, img.encode()
    
    def __len__(self):
        return len(self.img_list)
