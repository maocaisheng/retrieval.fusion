import os


class DbConfig(object):
    def __init__(self, root_dir, db_type, feature_dim, pretrained=True, split_num=3, noised=False, patch_num=1, network='senet154'):
        self.root_dir=root_dir
        self.db_type = db_type
        self.work_dir=os.path.join(root_dir, '{:s}/{:s}feature_{:s}_{:d}/{:d}'.format(
            db_type,
            'noised' if noised else '',
            'pretrain' if pretrained else 'metric',
            split_num,
            patch_num
        ))
        self.pca_data_dir = os.path.join(self.work_dir, 'pca_data')
        self.feature_dir = os.path.join(self.work_dir, 'results')
        self.db_feature=os.path.join(self.feature_dir, 'db_{:s}_{:d}.h5'.format(
            network,
            feature_dim
        ))
        self.query_feature=os.path.join(self.feature_dir, 'query_{:s}_{:d}.h5'.format(
            network,
            feature_dim
        ))
        self.save_dir = os.path.join(self.work_dir, 'query')
        self._check_exist(self.save_dir)
        self._check_exist(self.pca_data_dir)
        self._check_exist(self.feature_dir)

    def _check_exist(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)