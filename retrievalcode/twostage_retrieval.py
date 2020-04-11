import os
import faiss
import numpy as np
import math
from time import time
import networkx as nx
from tqdm import tqdm
from retrieval import gen_netflow_paths, read_feature, evaluate
from configs import DbConfig

def main():
    # read database
    DBTYPE='Copydays'
    feature_type = 'feats_MAC'
    feature_dim = 1024
    split_num = 5
    debug = False
#     add_noise_data = False

    if DBTYPE == 'Copydays':
        replace_ = '../db/Copydays/'
        topN = 20
        gt_file = '../db/Copydays/gt.csv'
    else:
        replace_ = '../YLS/'
        topN = 70
        gt_file = '../db/Synthdata/gt.csv'

    conf = DbConfig(root_dir='../ImageRetrieval', db_type=DBTYPE, feature_dim=feature_dim, pretrained=True, split_num=split_num, noised=False, patch_num=5)
    print(conf.db_feature)
    feats, feats_patch, db_name_list = read_feature(conf.db_feature, feature_type)
    feats_test, feats_patch_test, query_name_list = read_feature(conf.query_feature, feature_type)
    db_name_list = list(map(lambda x: x.replace(replace_, ''), db_name_list))
    query_name_list = list(map(lambda x: x.replace(replace_, ''), query_name_list))
#     if add_noise_data:
#         feats_noise, feats_patch_noise, noise_name_list = read_feature('../ImageRetrieval/Synthdata/feature_pretrain/soft/results/db_senet154_%d.h5' % feature_dim)
#         feats = np.concatenate((feats, feats_noise), axis=0)
#         feats_patch = np.concatenate((feats_patch, feats_patch_noise), axis=0)
#         db_name_list.extend(noise_name_list)

    patch_num=split_num**2
    feats_patch = feats_patch.reshape(-1, patch_num, feature_dim)
    feats_patch_test = feats_patch_test.reshape(-1, patch_num, feature_dim)    
    # coarse query
    index1 = faiss.IndexFlatL2(feats.shape[1])
    # gpu_index_flat = faiss.index_cpu_to_gpu(res,0,index_flat)
    print('DB shape: {}'.format(feats.shape))
    index1.add(feats)
    t = time()
    K = topN*3
    _, I1 = index1.search(feats_test, int(K))

    # final query
    results = []
    for qid in tqdm(range(len(query_name_list))):
        index2 = faiss.IndexFlatL2(feature_dim)
        index2.add(feats_patch[I1[qid]].reshape(-1, feature_dim))
        D2, I2 = index2.search(feats_patch_test[qid].reshape(-1, feature_dim), K)
        rank_fine = gen_netflow_paths(I2, D2, split_num, feature_dim, topK=K, neighbor=4)
        rank_fine = list(map(lambda x: I1[qid, x['img_id']], rank_fine))
        rank_coarse = list(I1[qid])
        candidate = rank_coarse + rank_fine
        candidate = list(set(candidate))
        score=[]
        for c in candidate:
            if c in rank_coarse:
                idx1 = rank_coarse.index(c)
            else:
                idx1 = len(rank_coarse)
            if c in rank_fine:
                idx2 = rank_fine.index(c)
            else:
                idx2 = len(rank_fine)
            tmp = 0.4*idx1+0.6*idx2
            score.append(tmp)
        newrank=filter(lambda x: x[1]<=topN, zip(candidate,score))
        #newrank = zip(candidate,score)
        newrank=sorted(newrank, key=lambda x: x[1])
        newrank=newrank[:min(len(newrank),topN)]
        if debug:
            print(query_name_list[qid])
            print('----')
            for i in range(len(newrank)):
                print(db_name_list[newrank[i][0]])
                print(newrank[i])
                print('----')
            print('>> path number:{}'.format(len(newrank)))
            break
        results.append(newrank)
    cost_time = time() - t
    # save result
    if not debug:
        save_file = os.path.join(conf.save_dir, '%s_dim%d_two_stage_query.csv' % (feature_type, feature_dim))
        with open(save_file, 'w+') as fid:
            for i in range(len(results)):
                line = [query_name_list[i]]
                for j in range(len(results[i])):
                    line += [db_name_list[results[i][j][0]]]
                fid.write(','.join(line) + '\n')
        print('write to: %s' % save_file)
        print('cost time: %.3f' % cost_time)
        evaluate(save_file, gt_file, topN, verbose=False, sep=',')


if __name__ == '__main__':
    main()
