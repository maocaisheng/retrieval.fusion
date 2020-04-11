import torchvision.transforms.functional as F
from torchvision.transforms import Lambda, Compose
import random
from PIL import Image

class DataAug(object):
    def __init__(self, resize, output_size, mean, std, istrain):
        self.resize = resize
        self.output_size = output_size
        self.mean = mean
        self.std = std
        self.istrain = istrain
        
    def __call__(self, img1, img2):
        if self.istrain:
            img1 = F.resize(img1, self.resize, Image.BILINEAR)
            img2 = F.resize(img2, self.resize, Image.BILINEAR)
            # random crop
            w1, h1 = img1.size
            w2, h2 = img2.size
            w = min(w1, w2)
            h = min(h1, h2)
            th, tw = self.output_size, self.output_size 
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            img1 = F.crop(img1, i, j, th, tw)
            img2 = F.crop(img2, i, j, th, tw)
            # random flip
            if random.random() < 0.5:
                img1 = F.hflip(img1)
                img2 = F.hflip(img2)
            if random.random() < 0.5:
                img1 = F.vflip(img1)
                img2 = F.vflip(img2)
            # color jitter
            brightness=0.4
            saturation=0.4
            hue=0.4
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            hue_factor = random.uniform(-hue, hue)
            transforms = []
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))
            random.shuffle(transforms)
            transform = Compose(transforms)
            img1 = transform(img1)
            img2 = transform(img2)
        else:
            img1 = F.resize(img1, (self.output_size, self.output_size), Image.BILINEAR)
            img2 = F.resize(img2, (self.output_size, self.output_size), Image.BILINEAR)
        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)
        img1 = F.normalize(img1, self.mean, self.std)
        img2 = F.normalize(img2, self.mean, self.std)
        return img1, img2