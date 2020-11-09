addpath('./FSLib_v4.2_2017/lib/')
addpath('./FSLib_v4.2_2017/methods/')
addpath('~/Escritorio/nuevo/FS/repo/gafs/c++/fitness/libsvm-3.20/matlab/')

%N_selected_features = 16; % madelon
%N_selected_features = 2; % leukemia
N_selected_features = 92; % GCM

%=============================
% LOAD TRAINING DATA
%====================
%data = csvread('data/madelon_TRAIN.csv');
%data = csvread('data/leukemia_38x7129_TRAIN.csv');
data = csvread('data/GCM_TRAIN.csv');

X_train = data(:,1:end-1);
Y_train = data(:,end);
Y_train(Y_train == -1) = 2;
clear data
%=============================
% LOAD TESTING DATA
%====================
%data = csvread('data/madelon_TEST.csv');
%data = csvread('data/leukemia_34x7129_TEST.csv');
data = csvread('data/GCM_TEST.csv');

X_test = data(:,1:end-1);
Y_test = data(:,end);
Y_test(Y_test == -1) = 2;
clear data
%=============================



% NUMBER OF PATTERNS AND FEATURES
[Npat, Nfeat] = size(X_train);



%==============================
% ALEATORIZO PATRONES
%=====================
idxs = randperm(Npat);
X_train = X_train(idxs,:);
Y_train = Y_train(idxs);


MIN = repmat(min(X_train),Npat,1);
MAX = repmat(max(X_train),Npat,1);
X_train = 2 * ( (X_train - MIN) ./ (MAX-MIN) - 0.5);

%trnD.x[i][k].value = 2.0*( (trnD.x[i][k].value - vmin[k]) / (vmax[k] - vmin[k]) - 0.5 );


% Select a feature selection method from the list
%listFS = {'InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};




% selection_method = 'relieff';
% selection_method = 'cfs';     % leukemia UAR: 0.5000, gcm UAR: 0.3750, madelon UAR: 0.5300
% selection_method = 'llcfs'; % gcm UAR: 0.4048, madelon UAR: 0.5083, leukemia UAR: 0.6893
selection_method = 'fisher';  % madelon UAR: 0.5350, leukemia UAR: 0.5000, GCM UAR: 0.4167
% selection_method = 'f0';  % GCM UAR: 0.4167, madelon UAR: 0.5333, leukemia UAR: 0.5000
% selection_method = 'laplacian'; % leukemia UAR: 0.5000, madelon UAR: 0.6067, GCM UAR: 0.3929

sprintf('Building features ranking...\n')

% feature Selection on training data
switch lower(selection_method)
    case 'mrmr'
        ranking = mRMR(X_train, Y_train, Nfeat);
        
    case 'relieff'
        [ranking, w] = reliefF( X_train, Y_train, 20);
        
    case 'mutinffs'
        [ ranking , w] = mutInfFS( X_train, Y_train, Nfeat );
        
    case 'fsv'
        [ ranking , w] = fsvFS( X_train, Y_train, Nfeat );
        
    case 'laplacian'
        W = dist(X_train');
        W = -W./max(max(W)); % it's a similarity
        [lscores] = LaplacianScore(X_train, W);
        [junk, ranking] = sort(-lscores);
        
    case 'mcfs'
        % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
        options = [];
        options.k = 5; %For unsupervised feature selection, you should tune
        %this parameter k, the default k is 5.
        options.nUseEigenfunction = 4;  %You should tune this parameter.
        [FeaIndex,~] = MCFS_p(X_train,Nfeat,options);
        ranking = FeaIndex{1};
        
    case 'rfe'
        ranking = spider_wrapper(X_train,Y_train,Nfeat,lower(selection_method));
        
    case 'l0'
        ranking = spider_wrapper(X_train,Y_train,Nfeat,lower(selection_method));
        
    case 'fisher'
        ranking = spider_wrapper(X_train,Y_train,Nfeat,lower(selection_method));
        
    case 'inffs'
        % Infinite Feature Selection 2015 updated 2016
        alpha = 0.5;    % default, it should be cross-validated.
        sup = 1;        % Supervised or Not
        [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );    
        
    case 'ecfs'
        % Features Selection via Eigenvector Centrality 2016
        alpha = 0.5; % default, it should be cross-validated.
        ranking = ECFS( X_train, Y_train, alpha )  ;
        
    case 'udfs'
        % Regularized Discriminative Feature Selection for Unsupervised Learning
        nClass = 14;
        ranking = UDFS(X_train , nClass ); % OUT OF MEMORY
        
    case 'cfs'
        % BASELINE - Sort features according to pairwise correlations
        ranking = cfs(X_train);     
        
    case 'llcfs'   
        % Feature Selection and Kernel Learning for Local Learning-Based Clustering
        ranking = llcfs( X_train );
        
    otherwise
        disp('Unknown method.')
end



% SELECCIONO LAS FEATURES QUE VOY A USAR
features = ranking(1:N_selected_features);

sprintf('Training SVM model...\n')
% ENTRENO EL MODELO
model = svmtrain(Y_train, X_train(:,features), '-s 0 -t 1 -c 10 -g 0.1 -r 1 -d 1 -q');


MIN = repmat(MIN(1,:),size(X_test,1),1);
MAX = repmat(MAX(1,:),size(X_test,1),1);
X_test = 2 * ( (X_test - MIN) ./ (MAX-MIN) - 0.5);

sprintf('Predicting Test outputs...\n')
% PREDIGO LOS DATOS
Y_predict = svmpredict(Y_test, X_test(:,features), model);



sprintf('Computing metrics...\n')
% CALCULO EL UAR
CP = classperf(Y_test, Y_predict);

UAR = sum((CP.SampleDistributionByClass - CP.ErrorDistributionByClass)./(numel(CP.ClassLabels)*CP.SampleDistributionByClass));

disp(sprintf('UAR: %0.4f',UAR));

