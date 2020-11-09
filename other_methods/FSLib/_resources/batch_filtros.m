

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




switch lower(selection_method)
    case 'sfs'
      % c = cvpartition(Y_train,'k',2);
      fun = @(XT,yT,Xt,yt)...
         (sum(~strcmp(yt,classify(Xt,XT,yT,'quadratic'))));
      % [fs,history] = sequentialfs(fun,X_train,Y_train,'cv',c,'direction','forward','nfeatures',N_selected_features)
      [fs,history] = sequentialfs(fun,X_train,Y_train,'direction','forward','nfeatures',N_selected_features)
        
    case 'sbs' 
      % c = cvpartition(Y_train,'k',2);
      fun = @(XT,yT,Xt,yt)...
         (sum(~strcmp(yt,classify(Xt,XT,yT,'quadratic'))));
      [fs,history] = sequentialfs(fun,X_train,Y_train,'direction','backward','nfeatures',N_selected_features)
end


switch lower(selection_method)
    case 'sfs'
       features = find(fs==1); 
    case 'sbs' 
       features = find(fs==1);
    otherwise
       % SELECCIONO LAS FEATURES QUE VOY A USAR
       features = ranking(1:N_selected_features);
end


% ENTRENO EL MODELO

% params = char('-s 0 -t 1 -c 10 -g 0.1 -r 1 -d 1 -q');
params = '-s 0 -t 1 -c 10 -g 0.1 -r 1 -d 1 -q';

model = svmtrain(Y_train, X_train(:,features), params);


MIN = repmat(MIN(1,:),size(X_test,1),1);
MAX = repmat(MAX(1,:),size(X_test,1),1);
X_test = 2 * ( (X_test - MIN) ./ (MAX-MIN) - 0.5);

% PREDIGO LOS DATOS
Y_predict = svmpredict(Y_test, X_test(:,features), model);


% CALCULO EL UAR
CP = classperf(Y_test, Y_predict);

UAR = sum((CP.SampleDistributionByClass - CP.ErrorDistributionByClass)./(numel(CP.ClassLabels)*CP.SampleDistributionByClass));

disp(sprintf('UAR: %0.4f',UAR));






