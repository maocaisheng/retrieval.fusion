import os
import math
import faiss
import numpy as np
from time import time
from retrieval import gen_netflow_paths, read_feature, evaluate
from configs import DbConfig
from tqdm import tqdm
import shutil
import torchvision

def main():
    # read database
    DBTYPE='Copydays'
    feature_type = 'feats_MAC'
    feature_dim = 1024
    split_num = 3
    debug = True
#     add_noise_data = False

    if DBTYPE == 'Copydays':
        replace_ = '../db/Copydays/'
        topN = 20
        gt_file = '../db/Copydays/gt.csv'
    else:
        replace_ = '../YLS/'
        topN = 70
        gt_file = '../db/Synthdata/gt.csv'

    conf = DbConfig(root_dir='../ImageRetrieval', db_type=DBTYPE, feature_dim=feature_dim, pretrained=True, split_num=split_num, noised=False)
    print(conf.db_feature)
    _, feats_patch, db_name_list = read_feature(conf.db_feature, feature_type)
    _, feats_patch_test, query_name_list = read_feature(conf.query_feature, feature_type)
    db_name_list = list(map(lambda x: x.replace(replace_, ''), db_name_list))
    query_name_list = list(map(lambda x: x.replace(replace_, ''), query_name_list))
#     if add_noise_data:
#         _, feats_patch_noise, noise_name_list = read_feature('../ImageRetrieval/Synthdata/feature_pretrain/soft/results/db_senet154_%d.h5' % feature_dim, feature_type)
#         feats_patch = np.concatenate((feats_patch, feats_patch_noise), axis=0)
#         db_name_list.extend(noise_name_list)
    print(feats_patch.shape)

    # final query
    t = time()
    index = faiss.IndexFlatL2(feature_dim)
    index.add(feats_patch)
    K = topN*3
    _, I = index.search(feats_patch_test, K)
    patch_num = split_num**2
    results = []

    for qid in tqdm(range(len(query_name_list))):
        qid = query_name_list.index('copydays_original/202500.jpg')
        top = np.empty((patch_num, K), np.int32)
        id_set = set()
        for pid in range(patch_num):
            for tid in range(K):
                img_id = I[qid * patch_num + pid, tid] // patch_num  # image_ID
                id_set.add(img_id)
                top[pid, tid] = img_id

        votes = []
        id_set = list(id_set)
        for img_id in id_set:
            v = 0
            for pid in range(patch_num):
                if img_id in list(top[pid]):
                    v += 1
            votes.append(v)
        votes = list(zip(id_set, votes))
        votes.sort(key=lambda x: -x[1])
        results.append(votes[:min(len(votes), topN)])
        #print('%s: %d'%(query_name_list[qid], len(results[-1])))
        if debug:
            from PIL import Image
            import matplotlib.pyplot as plt
            print(query_name_list[qid])
            print('----')
            imgs=[]
            trans = [torchvision.transforms.Resize((200,200)), torchvision.transforms.ToTensor()]
            trans = torchvision.transforms.Compose(trans)
            for i in range(10):
                img = Image.open(os.path.join(replace_, db_name_list[votes[i][0]]))
                img = trans(img)
                imgs.append(img)
                print(db_name_list[votes[i][0]])
                print(votes[i])
            vis=torchvision.utils.make_grid(imgs, nrow=5)
            vis = vis.numpy().transpose((1,2,0))
            plt.imsave('region.png',vis, format="png")
            print(vis.shape)
            plt.imshow(vis)
            plt.show()
            print('>> candidate image number:{}'.format(len(votes)))
            break
        
    cost_time = time() - t
    # save result
    if not debug:
        save_file = os.path.join(conf.save_dir, '%s_dim%d_region_query.csv' % (feature_type, feature_dim))
        with open(save_file, 'w+') as fid:
            for i in range(len(results)):
                line = [query_name_list[i]]
                for j in range(len(results[i])):                 
                    line += [db_name_list[results[i][j][0]]]
                fid.write(','.join(line) + '\n')
        print('>> write to: %s' % save_file)
        print('cost time: %.3f' % cost_time)
        evaluate(save_file, gt_file, topN, verbose=True, sep=',')


if __name__ == '__main__':
    main()
