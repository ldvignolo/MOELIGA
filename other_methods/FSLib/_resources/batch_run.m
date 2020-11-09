% addpath('./FSLib_v4.2_2017/lib/')
% addpath('./FSLib_v4.2_2017/methods/')
% addpath('~/Escritorio/nuevo/FS/repo/gafs/c++/fitness/libsvm-3.20/matlab/')

addpath('./FSLib_v7.0.1_2020_2/lib/')
addpath('./FSLib_v7.0.1_2020_2/methods/')
addpath('./libsvm-3.24/matlab/')

N_selected_features = 19; % madelon

%=============================
% LOAD TRAINING DATA
%====================
data = csvread('data/madelon_TRAIN.csv');

X_train = data(:,1:end-1);
Y_train = data(:,end);
Y_train(Y_train == -1) = 2;
clear data
%=============================
% LOAD TESTING DATA
%====================
data = csvread('data/madelon_TEST.csv');

X_test = data(:,1:end-1);
Y_test = data(:,end);
Y_test(Y_test == -1) = 2;
clear data
%=============================

selection_method = 'cfs';       
disp(' '); disp('madelon - cfs');
batch_filtros
selection_method = 'llcfs';
disp(' '); disp('madelon - llcfs');
batch_filtros
selection_method = 'fisher';    
disp(' '); disp('madelon - fisher');
batch_filtros
selection_method = 'laplacian'; 
disp(' '); disp('madelon - laplacian');
batch_filtros
selection_method = 'rfe';
disp(' '); disp('madelon - RFE');
batch_filtros
selection_method = 'relieff';
disp(' '); disp('madelon - relefF');
batch_filtros


% ---------------------------------------------------

N_selected_features = 2; % leukemia

%=============================
% LOAD TRAINING DATA
%====================
data = csvread('data/leukemia_38x7129_TRAIN.csv');

X_train = data(:,1:end-1);
Y_train = data(:,end);
Y_train(Y_train == -1) = 2;
clear data
%=============================
% LOAD TESTING DATA
%====================
data = csvread('data/leukemia_34x7129_TEST.csv');

X_test = data(:,1:end-1);
Y_test = data(:,end);
Y_test(Y_test == -1) = 2;
clear data
%=============================

selection_method = 'cfs';       
disp(' '); disp('leukemia - cfs');
batch_filtros
selection_method = 'llcfs';
disp(' '); disp('leukemia - llcfs');
batch_filtros
selection_method = 'fisher';    
disp(' '); disp('leukemia - fisher');
batch_filtros
selection_method = 'laplacian'; 
disp(' '); disp('leukemia - laplacian');
batch_filtros
selection_method = 'rfe';
disp(' '); disp('leukemia - RFE');
batch_filtros
selection_method = 'relieff';
disp(' '); disp('leukemia - relefF');
batch_filtros



% ----------------------------------------------------

N_selected_features = 92; % GCM

%=============================
% LOAD TRAINING DATA
%====================
data = csvread('data/GCM_TRAIN.csv');

X_train = data(:,1:end-1);
Y_train = data(:,end);
Y_train(Y_train == -1) = 2;
clear data
%=============================
% LOAD TESTING DATA
%====================
data = csvread('data/GCM_TEST.csv');

X_test = data(:,1:end-1);
Y_test = data(:,end);
Y_test(Y_test == -1) = 2;
clear data
%=============================


selection_method = 'cfs';       
disp(' '); disp('GCM - cfs');
batch_filtros
selection_method = 'llcfs';
disp(' '); disp('GCM - llcfs');
batch_filtros
selection_method = 'fisher';    
disp(' '); disp('GCM - fisher');
batch_filtros
selection_method = 'laplacian'; 
disp(' '); disp('GCM - laplacian');
batch_filtros
selection_method = 'rfe';
disp(' '); disp('GCM - RFE');
batch_filtros
selection_method = 'relieff';
disp(' '); disp('GCM - relefF');
batch_filtros
