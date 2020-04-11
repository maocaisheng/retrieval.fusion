clear;
%clc;

addpath('vlfeat-0.9.21/toolbox');
run('vl_setup');

%% db_type, TRAIN, PRETRAIN, cluster_num

%feature_vlad('Copydays', 1, 1, 256);
%feature_vlad('Copydays', 0, 1, 256);

%feature_vlad('Copydays', 1, 0, 256);
%feature_vlad('Copydays', 0, 0, 256);

%feature_vlad('Challenging', 1, 1, 256);
%feature_vlad('Challenging', 0, 1, 256);

%feature_vlad('Challenging', 1, 0, 256);
%feature_vlad('Challenging', 0, 0, 256);

%feature_vlad('Copydays', 1, 0, 512);
%feature_vlad('Copydays', 0, 0, 512);


%feature_vlad('Copydays', 1, 1, 32);
feature_vlad('Copydays', 0, 1, 32);

%feature_vlad('Copydays', 1, 1, 64);
feature_vlad('Copydays', 0, 1, 64);

%feature_vlad('Copydays', 1, 1, 128);
feature_vlad('Copydays', 0, 1, 128);

%feature_vlad('Copydays', 1, 1, 256);
feature_vlad('Copydays', 0, 1, 256);

%feature_vlad('Copydays', 1, 1, 512);
feature_vlad('Copydays', 0, 1, 512);

%feature_vlad('Copydays', 1, 1, 1024);
feature_vlad('Copydays', 0, 1, 1024);