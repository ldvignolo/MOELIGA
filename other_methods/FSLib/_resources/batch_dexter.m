addpath('./FSLib_v4.2_2017/lib/')
addpath('./FSLib_v4.2_2017/methods/')
addpath('~/Escritorio/nuevo/FS/repo/gafs/c++/fitness/libsvm-3.20/matlab/')

%N_selected_features = 136; % dexter


%  %=============================
%  % LOAD TRAINING DATA
%  %====================
%  [dataName,attributeName, attributeType, data]= arffread('data/dexter_TRAIN.arff');
%  
%  X_train = data(:,1:end-1);
%  Y_train = data(:,end);
%  Y_train(Y_train == -1) = 2;
%  clear data
%  %=============================
%  % LOAD TESTING DATA
%  %====================
%  [dataName,attributeName, attributeType, data]= arffread('data/dexter_TEST.arff');
%  
%  X_test = data(:,1:end-1);
%  Y_test = data(:,end);
%  Y_test(Y_test == -1) = 2;
%  clear data
%  %=============================



%  selection_method = 'fisher';    
%  disp(' '); disp('dexter - fisher');
%  batch_filtros
%  save('dexter_fisher.pred','Y_predict','-ascii')
%  selection_method = 'laplacian'; 
%  disp(' '); disp('dexter - laplacian');
%  batch_filtros
%  save('dexter_laplacian.pred','Y_predict','-ascii')
%  selection_method = 'cfs';       
%  disp(' '); disp('dexter - cfs');
%  batch_filtros
%  save('dexter_cfs.pred','Y_predict','-ascii')
selection_method = 'llcfs';
disp(' '); disp('dexter - llcfs');
batch_filtros
save('dexter_llcfs.pred','Y_predict','-ascii')


