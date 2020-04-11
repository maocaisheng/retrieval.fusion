import os
import faiss
import numpy as np
from time import time
from retrieval import evaluate
from configs import DbConfig
import torchvision
import h5py
import copy
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def read_feature(name):
    print(name)
    h5f = h5py.File(name,'r')
    feats = h5f['feats'][:]
    name_list = h5f['name_list'][:]
    h5f.close()
    return feats, name_list

def read_mat_feature(name):
    print(name)
    from scipy.io import loadmat
    m = loadmat(name)
    feats = m['dataset_features'].astype(np.float32)
    name_list = []
    for i in range(m['name_list'].shape[0]):
        name_list.extend(m['name_list'][i][0].tolist())
    return feats, name_list

def main(db_type):
    # read database
    # feature_dim = 1024
#     add_noise_data = False

    feature_type='CONV4_RMAC'
    if db_type == 'Copydays':
        replace_ = '../db/Copydays/'
        topN = 20
        gt_file = '../db/Copydays/gt.csv'
    else:
        replace_ = '../db/Challenging/'
        topN = 15
        gt_file = '../db/Challenging/gt.csv'

    work_dir = 'features/%s/feature_{:s}/'%(db_type)
    save_dir = os.path.join(work_dir.format('fuse'),'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    db_feature_conv4 = os.path.join(work_dir.format('pretrain'),'feat','CONV4_db_cluster_256.mat')
    query_feature_conv4 = os.path.join(work_dir.format('pretrain'),'feat','CONV4_query_cluster_256.mat')
    feats_conv, db_name_list = read_mat_feature(db_feature_conv4)
    feats_test_conv, query_name_list = read_mat_feature(query_feature_conv4)
    db_feature = os.path.join(work_dir.format('metric'),'feat','RMAC_db_pca_1024.h5')
    query_feature=os.path.join(work_dir.format('metric'),'feat','RMAC_query_pca_1024.h5')
    feats_rmac, _ = read_feature(db_feature)
    feats_test_rmac, _ = read_feature(query_feature)  
    
    feats = np.concatenate((feats_conv, feats_rmac), axis=1)
    feats_test = np.concatenate((feats_test_conv, feats_test_rmac), axis=1)
    pca=PCA(n_components=1024, whiten=False)
    pca.fit(feats)
    feats=pca.transform(feats)
    feats_test= pca.transform(feats_test)
    db_name_list = list(map(lambda x: x.replace(replace_, ''), db_name_list))
    query_name_list = list(map(lambda x: x.replace(replace_, ''), query_name_list))
    
#     print(db_name_list[:10])
#     print(query_name_list[:10])
#     print(feats.shape)
#     print(feats.dtype)
    # feats=deepcopy(feats)
    
    feats=feats.copy()
    feats_test=feats_test.copy()
    
    #DBA
    t1=time()
#     DBA_num = 2
#     #res = faiss.StandardGpuResources()
#     index_flat = faiss.IndexFlatL2(feats.shape[1])
#     #gpu_index_flat = faiss.index_cpu_to_gpu(res,0,index_flat)
#     feats=feats.astype('float32')
#     index_flat.add(feats)
#     D,I = index_flat.search(feats,DBA_num)
    
#     new_feats = copy.deepcopy(feats)
#     for num in range(len(I)):
#         new_feat = feats[I[num][0]]
#         for num1 in range(1,len(I[num])):
#             weight = (len(I[num])-num1) / float(len(I[num]))
#             new_feat += feats[num1] * weight
#         new_feats[num]=new_feat

    new_feats=feats
    #QE
#     QE_num = 2
#     #res = faiss.StandardGpuResources()
#     index_flat = faiss.IndexFlatL2(new_feats.shape[1])
#     #gpu_index_flat = faiss.index_cpu_to_gpu(res,0,index_flat)
#     feats_test=feats_test.astype('float32')
#     index_flat.add(new_feats)
#     D,I = index_flat.search(feats_test,QE_num - 1)
#     query_feat=copy.deepcopy(feats_test)
#     for num in range(len(query_feat)):
#         query_feat[num]=(query_feat[num]+new_feats[I.T[0,num]]) / float(QE_num)   
    
    
    query_feat=feats_test
    # final query
    index_flat = faiss.IndexFlatL2(new_feats.shape[1])
    #gpu_index_flat = faiss.index_cpu_to_gpu(res,0,index_flat)
    index_flat.add(new_feats)
    D,I = index_flat.search(query_feat,topN)
    t2 = time()
    cost_time = t2-t1
    
    # save result
    save_file = os.path.join(save_dir, '%s_query.csv'%(feature_type))
    with open(save_file, 'w+') as fid:
        for i in range(I.shape[0]):
            line=[query_name_list[i]]
            for j in range(I.shape[1]):
                line += [db_name_list[I[i,j]]]
            fid.write('\t'.join(line)+'\n')
    print('write to: %s'%save_file)
    print('cost time: %.3f'%cost_time)
    evaluate(save_file, gt_file, topN, verbose=False, sep='\t')
    
    i = query_name_list.index('296_1.jpg')#344_1.jpg 457_1.jpg
    line = [query_name_list[i]]
    imgs=[]
    trans = [torchvision.transforms.Resize((200,200)), torchvision.transforms.ToTensor()]
    trans = torchvision.transforms.Compose(trans)
    for j in range(1,11):
        #print(db_name_list[I[i,j]])
        name = db_name_list[I[i,j]]
        if j>5:
            img = Image.open(os.path.join(replace_,'CopyAttack', '287', '287_1-%02d.jpg'%j))
        else:
            img = Image.open(os.path.join(replace_,'GroundTruth', '287', '287_%d.jpg'%j))
        img = trans(img)
        imgs.append(img)
    vis=torchvision.utils.make_grid(imgs, nrow=5)
    vis = vis.numpy().transpose((1,2,0))
    plt.imsave('%s.png'%feature_type,vis, format="png")
    print(vis.shape)
    plt.imshow(vis)
    plt.show()

if __name__ == '__main__':
    main('Challenging')
