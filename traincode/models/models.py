import torch
from torch.utils import model_zoo
# from pretrainedmodels.models.senet import SENet, SEBottleneck
from .senet import SENet,SEBottleneck
from .resnext import ResNeXt101_64x4d
default_settings = {
    'url': {
        'senet154': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
        'resnext101_64x4d': 'http://data.lip6.fr/cadene/pretrainedmodels/resnext101_64x4d-e77a0586.pth',
    },
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

def load_state_dict(model, model_url):
    model.mean = default_settings['mean']
    model.std = default_settings['std']
    if model_url.startswith('http'):
        pretrained_dict = model_zoo.load_url(model_url)
    else:
        pretrained_dict = torch.load(model_url)['MODEL'] # NOTATION !!!! see save model
        print('Loading from LOCAL weights!!')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def SENet154(pretrained=False):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2)
    if pretrained:
        model_url = default_settings['url']['senet154']
        load_state_dict(model, model_url)
    return model

def ResNeXt101(pretrained=False):
    model = ResNeXt101_64x4d()
    if pretrained:
        model_url = default_settings['url']['resnext101_64x4d']
        load_state_dict(model, model_url)
    return model
    
    