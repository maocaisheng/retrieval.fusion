import h5py


def read_feature(h5file, feats_type):
    h5f = h5py.File(h5file, 'r')
    feats_patch = h5f[feats_type]['feats_patch'][:]
    feats_img = h5f[feats_type]['feats_img'][:]
    name_list = [x.decode() for x in h5f['name_list'][:]]
    h5f.close()
    return feats_img, feats_patch, name_list
