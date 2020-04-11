import os
import copy
import faiss
import numpy as np
import math
from time import time
import matplotlib.pyplot as plt
from retrieval import gen_netflow_paths, read_feature, evaluate
from configs import DbConfig
from tqdm import tqdm
import torchvision
import shutil
np.random.seed(2019)

def main():
    # read database
    DBTYPE='Copydays'
    feature_type = 'feats_MAC'
    feature_dim = 1024
    split_num = 3
    debug = False
#     add_noise_data = True

    if DBTYPE == 'Copydays':
        replace_ = '../db/Copydays/'
        topN = 20
        gt_file = '../db/Copydays/gt.csv'
    else:
        replace_ = '../YLS/'
        topN = 70
        gt_file = '../db/Synthdata/gt.csv'

    conf = DbConfig(root_dir='../ImageRetrieval', db_type=DBTYPE, feature_dim=feature_dim, pretrained=True, split_num=split_num, noised=False, patch_num=20)
    print(conf.db_feature)
    _, feats_patch, db_name_list = read_feature(conf.db_feature, feature_type)
    _, feats_patch_test, query_name_list = read_feature(conf.query_feature, feature_type)
    db_name_list = list(map(lambda x: x.replace(replace_, ''), db_name_list))
    query_name_list = list(map(lambda x: x.replace(replace_, ''), query_name_list))
#     if add_noise_data:
#         _, feats_patch_noise, noise_name_list = read_feature('../ImageRetrieval/Imagenet/feature_pretrain/soft/feat_db_senet154_1.h5', feature_type)
#         feats_patch = np.concatenate((feats_patch, feats_patch_noise), axis=0)
#         db_name_list.extend(noise_name_list)
    print(feats_patch.shape)

    # final query
    t = time()
    index = faiss.IndexFlatL2(feature_dim)
    index.add(feats_patch)
    K = topN * 3
    D, I = index.search(feats_patch_test, K)
    results = []

    patch_num = split_num**2
    I = I.reshape(-1, patch_num, K)
    D = D.reshape(-1, patch_num, K)
    # print(D[0])
    for qid in tqdm(range(len(query_name_list))):
        #qid = query_name_list.index('copydays_original/202500.jpg')

        rank = gen_netflow_paths(I[qid], D[qid], split_num, feature_dim, topK=K, neighbor=4)
        rank = rank[:min(len(rank), topN)]
        #print(len(rank))
        results.append(rank)
        if debug:
            from PIL import Image
            import matplotlib.pyplot as plt            
            print('----')
#             imgs=[]
#             trans = [torchvision.transforms.Resize((200,200)), torchvision.transforms.ToTensor()]
#             trans = torchvision.transforms.Compose(trans)
            for i in range(10):
#                 pth = os.path.join(replace_, db_name_list[rank[i]['img_id']])
#                 img = Image.open(pth)
#                 shutil.copyfile(pth,'%d.jpg'%i)
#                 img = trans(img)
#                 imgs.append(img)
                print(db_name_list[rank[i]['img_id']])
                print(rank[i])
                print('----')
#             vis=torchvision.utils.make_grid(imgs, nrow=5)
#             vis = vis.numpy().transpose((1,2,0))
#             plt.imsave('patch.png',vis, format="png")
#             print(vis.shape)
#             plt.imshow(vis)
#             plt.show()                
            print('>> path number:{}'.format(len(rank)))
            break

    cost_time = time() - t
    # save result
    if not debug:
        save_file = os.path.join(conf.save_dir, '%s_dim%d_patch_query.csv' % (feature_type, feature_dim))
        with open(save_file, 'w+') as fid:
            for i in range(len(results)):
                line = [query_name_list[i]]
                for j in range(len(results[i])):
                    line += [db_name_list[results[i][j]['img_id']]]
                fid.write(','.join(line) + '\n')
        print('write to: %s' % save_file)
        print('cost time: %.3f' % cost_time)
        evaluate(save_file, gt_file, topN, verbose=False, sep=',')


if __name__ == '__main__':
    main()
