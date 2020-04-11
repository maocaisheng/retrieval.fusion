import os
import h5py
import math
from imageretrievalnet import init_network, extract_vectors
from easydict import EasyDict

GPU_ID = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


def save_feature(images, model_choose, image_size, feature_path, pretrain, isquery):
    model = init_network(model_choose, pretrain)
    sub_set_num = 20000
    image_num = len(images)
    split_num = math.ceil(image_num / sub_set_num)

    print('>>processing {} images\nsplit to {} subsets'.format(image_num, split_num))
    for s in range(split_num):
        images_sub = images[sub_set_num * s: min(sub_set_num * (s + 1), image_num)]
        vecs_RMAC, mats_conv4, name_list = extract_vectors(model, images_sub, image_size)
        #feats_MAC = vecs_MAC.numpy()
        #feats_SPoC = vecs_SPoC.numpy()
        feats_RMAC = vecs_RMAC.numpy()
        mats_conv4 = mats_conv4.numpy()
        #feats_RAMAC = vecs_RAMAC.numpy()
        print('features shape: {}'.format(mats_conv4.shape))
        if isquery == 0:
            name = os.path.join(feature_path, 'feat_db_{:s}_{:d}.h5'.format(model_choose, s + 1))
        else:
            name = os.path.join(feature_path, 'feat_query_{:s}_{:d}.h5'.format(model_choose, s + 1))
        h5f = h5py.File(name, 'w')
        h5f.create_dataset('feats_RMAC', data=feats_RMAC)
        h5f.create_dataset('mats_conv4', data=mats_conv4)
        h5f.create_dataset('name_list', data=name_list)
        h5f.close()
        print('\r>>>> save to {}.'.format(name))

def main(db, model_choose, pretrain = True):
    print('=============')
    print(db.upper())
    print('=============')
    save_dir = 'features/{:s}/feature_{:s}/'.format(db, 'pretrain' if pretrain else 'metric')
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    images_db = []
    images_query = set()
    root_dir = '../db/%s/'%db
    with open(os.path.join(root_dir, 'db.txt'), 'r') as f:
        images_db = [os.path.join(root_dir, x.strip()) for x in f.readlines()]
    with open(os.path.join(root_dir, 'query.txt'), 'r') as f:
        images_query = [os.path.join(root_dir, x.strip()) for x in f.readlines()]
    save_feature(images_db, model_choose, 224, save_dir, pretrain, isquery=False)
    save_feature(images_query, model_choose, 224, save_dir, pretrain, isquery=True)


if __name__ == '__main__':
    main('Copydays', model_choose='resnext101', pretrain = False)
    #main('Challenging', model_choose='resnext101', pretrain = True)
    # synthdata(False)
    # imagenet(pretrain=True, hard_split=False)
    # holidays(pretrain=True, hard_split=False)

