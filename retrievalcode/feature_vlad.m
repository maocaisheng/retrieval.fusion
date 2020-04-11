function feature_vlad(db_type, TRAIN, PRETRAIN, cluster_num)

local_feature_dim=128;
global_feature_dim=1024;

if PRETRAIN
    work_dir = fullfile('features', db_type, 'feature_pretrain');
else
    work_dir = fullfile('features', db_type, 'feature_metric');
end
save_dir = fullfile(work_dir, 'feat');
pca_data_dir =fullfile(work_dir, 'pca_dat');
if TRAIN
    name='feat_db_resnext101_1.h5';
else
    name='feat_query_resnext101_1.h5';
end    
feature_file=fullfile(work_dir, name)
features=h5read(feature_file,'/mats_conv4'); %14x14x1024x3055
name_list=h5read(feature_file,'/name_list');
[W,H,feat_dim,N] = size(features);
%disp(size(features))
features=permute(features,[3,1,2,4]);
features=reshape(features, feat_dim, []);

if ~TRAIN
    D=load(fullfile(pca_data_dir, sprintf('conv4_cluster_%d.mat',cluster_num)));
end

features=double(features);
%size(features)
%disp('performing PCA local feature dim reduction');
%L2 Normalized
for i=1:size(features,2)
    features(:,i)=features(:,i) / (norm(features(:,i),2) + 1e-5);
end
%PCA and Whiting
if TRAIN
    mean_local=mean(features,2);
    features= bsxfun(@minus, features, mean_local);
    [coeff_local,~,latent_local]=pca(features','NumComponents',local_feature_dim,'Centered',false);
else
    coeff_local=D.coeff_local;
    latent_local=D.latent_local;
    features= bsxfun(@minus, features, D.mean_local);
end
cs1=cumsum(latent_local)/sum(latent_local);
fprintf('first PCA: %.4f\n', cs1(128))
features=coeff_local'*features;
for i=1:size(features,1)
    features(i,:)=features(i,:) / (sqrt(latent_local(i) + 1e-10));
end
%L2 Normalized
for i=1:size(features,2)
    features(:,i)=features(:,i) / (norm(features(:,i),2) + 1e-5);
end
if TRAIN
    centers = vl_kmeans(features, cluster_num, 'Initialization', 'plusplus');
else
    centers=D.centers;
end
kdtree = vl_kdtreebuild(centers);
features=reshape(features,size(features,1),[],N);

%disp('VLAD encoding');
is_first=1;
for i=1:N
    nn = vl_kdtreequery(kdtree, centers, features(:,:,i));
    assignments = zeros(cluster_num,size(features,2));
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    if is_first == 1
        dataset_features=zeros(local_feature_dim * cluster_num, N);
        is_first=0;
    end
    dataset_features(:,i) = vl_vlad(features(:,:,i),centers,assignments,'NormalizeComponents');%Already L2 Normalized
end

%disp('performing PCA global feature dim reduction');
%PCA and Whiting
if TRAIN
    mean_global=mean(dataset_features,2);
    dataset_features= bsxfun(@minus, dataset_features, mean_global);
    [coeff_global,~,latent_global]=pca(dataset_features','NumComponents',global_feature_dim,'Centered',false); % 'Algorithm','eig'
else
    coeff_global=D.coeff_global;
    latent_global=D.latent_global;
    dataset_features= bsxfun(@minus, dataset_features, D.mean_global);
end
cs2=cumsum(latent_global)/sum(latent_global);
fprintf('second PCA: %.4f\n',cs2(1024))
dataset_features=coeff_global'*dataset_features;
for i=1:size(dataset_features,1)
    dataset_features(i,:)=dataset_features(i,:) / sqrt(latent_global(i) + 1e-10);
end
%L2 Normalized
for i=1:size(dataset_features,2)
    dataset_features(:,i)=dataset_features(:,i) / (norm(dataset_features(:,i),2) + 1e-5);
end

dataset_features=single(dataset_features');

if TRAIN
    stage='db';
else
    stage='query';
end
%save_file_name=fullfile(save_dir,sprintf('CONV4_%s_cluster_%d.mat', stage, cluster_num))
%save(save_file_name,'dataset_features','name_list');
if TRAIN
    save(fullfile(pca_data_dir, sprintf('conv4_cluster_%d.mat',cluster_num)),'centers','coeff_local','latent_local','mean_local','coeff_global','latent_global','mean_global');
end
disp('all is done!');
end