import h5py
import pickle
from sklearn.decomposition import PCA
import numpy as np
import time
import os
from configs import DbConfig

def read_feature(name):
    h5f = h5py.File(name,'r')
    #feats_MAC = h5f['feats_MAC'][:]
    #feats_SPoC = h5f['feats_SPoC'][:]
    feats_RMAC = h5f['feats_RMAC'][:]
    mats_conv4 = h5f['mats_conv4'][:]
    name_list = h5f['name_list'][:]
    print(name_list[0])
    h5f.close()
    return feats_RMAC, mats_conv4, name_list
def check_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)
        
def L2Norm(feature):
    eps = 1e-9
    l2_norm = np.linalg.norm(feature, axis=1, keepdims=True)
    #print(l2_norm.shape)
    feature = feature / (l2_norm + eps)
    return feature
    
def main(db_type, isquery, pretrain=True):
    assert(db_type in ['Synthdata', 'Copydays', 'Challenging'])

    pca_dim = 1024
    work_dir = 'features/{:s}/feature_{:s}/'.format(db_type, 'pretrain' if pretrain else 'metric')
    pca_data_dir=os.path.join(work_dir, 'pca_dat')
    save_dir = os.path.join(work_dir, 'feat')
    check_dir(pca_data_dir)
    check_dir(save_dir)
    print(work_dir)
    if isquery:
        file_name = 'feat_query_resnext101_1.h5'
    else:
        file_name = 'feat_db_resnext101_1.h5'
    feats_RMAC, _, name_list = read_feature(os.path.join(work_dir, file_name))
    
    feature_type='RMAC'
    print('>> start processing...{}'.format(feature_type))
    feature = feats_RMAC
    image_num, feature_dim = feature.shape
    print('>> feature shape: {}'.format(feature.shape))

    t1 = time.time()
    if isquery:
        with open(os.path.join(pca_data_dir, '{}_dim_{}.pkl'.format(feature_type, pca_dim)), 'rb') as f:
            pca = pickle.load(f)
    else:
        pca=PCA(n_components=pca_dim, whiten=False)
        pca.fit(feature)
        with open(os.path.join(pca_data_dir, '{}_dim_{}.pkl'.format(feature_type, pca_dim)), 'wb') as f:
            pickle.dump(pca, f, protocol=4)
            
    feature = pca.transform(feature)
    feature = L2Norm(feature)
    t2 = time.time()
    print('>> PCA finished. ({:.1f}s)'.format(t2-t1))

    save_path = os.path.join(save_dir, '{:s}_{:s}_pca_{:d}.h5'.format(feature_type, 'query' if isquery else 'db', pca_dim))
    h5f = h5py.File(save_path, 'w')
    h5f.create_dataset('feats', data = feature)
    h5f.create_dataset('name_list', data = name_list)
    h5f.close()
    print('>> save to {}.'.format(save_path))

if __name__=='__main__':
    main(db_type='Copydays', isquery = False, pretrain=False)
    main(db_type='Copydays', isquery = True, pretrain=False)
#     main(db_type='Synthdata', isquery = True, hard_split = False)
#     main(db_type='Synthdata', isquery = False, hard_split = True)
#     main(db_type='Synthdata', isquery = True, hard_split = True)