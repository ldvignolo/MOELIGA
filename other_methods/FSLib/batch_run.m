clear; clc;

addpath('./FSLib_v7.0.1_2020_2/lib/')
addpath('./FSLib_v7.0.1_2020_2/methods/')
addpath('./libsvm-3.24/matlab/')
addpath('./jsonlab-2.0/')



FSMethods = {'cfs', 'fisher','laplacian', 'rfe','relieff','mutinffs','fsv','mcfs','L0', 'UDFS','llcfs','InfFS','ECFS','mrmr'};
%  indices   = [    1,        1,          0,     0,        1,         0,    0,     0,   0,      0,      0,      0,     0,     0];
indices   = [    0,        0,          0,     0,        1,         0,    0,     0,   0,      0,      0,      0,     0,     0];

datasets  = {'madelon', 'leuk', 'gcm', 'gisette'};
%  dbi       = [        0,      0,     0,        1 ] ;
dbi       = [        0,      1,     0,        0 ] ; 


FSMethods = FSMethods(find(indices));

rfK = 20;  % ReliefF K parameter

Repes = 4;

for i=1:Repes,
    
    rng('shuffle');

    if (dbi(1)),
        
        [X_train, Y_train] = arffRead('data/madelon.trn.arff');
    
        idx = randperm(length(Y_train));
        X_train = X_train(idx,:);
        Y_train = Y_train(idx);

        N_selected_features = 61; % madelon
        dataset='madelon';
        for i=1:length(FSMethods),
            % rfK = 20*i;
            selection_method = char(FSMethods(i));
            disp(strcat(dataset, {' - '}, selection_method));
            batch_filtros
        end;
        % disp( features(1:min(length(features),10)));
        T = cell2table(Table(2:end,:),'VariableNames',Table(1,:))
        writetable(T, ['_resultados/' dataset '.csv'])
        clear Table features ranking;
    end

        
    if (dbi(2)),
        
        [X_train, Y_train] = arffRead('data/leukemia_train_38x7129.arff');
    
        idx = randperm(length(Y_train));
        X_train = X_train(idx,:);
        Y_train = Y_train(idx);
    
        N_selected_features = 131; % leukemia
        dataset='leuk';
        for i=1:length(FSMethods),
            selection_method = char(FSMethods(i));
            disp(strcat(dataset, {' - '}, selection_method));
            batch_filtros
        end;
        % disp(features(1:min(length(features),10)));
        T = cell2table(Table(2:end,:),'VariableNames',Table(1,:))
        writetable(T, ['_resultados/' dataset '.csv'])
        clear Table features ranking;
    
    end


    if (dbi(3)),
        
        [X_train, Y_train] = arffRead('data/GCM_Training.arff');
    
        idx = randperm(length(Y_train));
        X_train = X_train(idx,:);
        Y_train = Y_train(idx);    
    
        N_selected_features = 587; % GCM
        dataset='gcm';

        for i=1:length(FSMethods),
            selection_method = char(FSMethods(i));
            disp(strcat(dataset, {' - '}, selection_method));
            batch_filtros
        end;
        % disp(features(1:min(length(features),10)));
        T = cell2table(Table(2:end,:),'VariableNames',Table(1,:))
        writetable(T, ['_resultados/' dataset '.csv'])
        clear Table features ranking;
    
    end

    if (dbi(4)),
        
        [X_train, Y_train] = arffRead('data/Gisette/gisette_train.arff');

        idx = randperm(length(Y_train));
        X_train = X_train(idx,:);
        Y_train = Y_train(idx);    
    
        N_selected_features = 50;
        dataset='gisette';

        for i=1:length(FSMethods),
            selection_method = char(FSMethods(i));
            disp(strcat(dataset, {' - '}, selection_method));
            batch_filtros
        end;
        % disp(features(1:min(length(features),10)));
        T = cell2table(Table(2:end,:),'VariableNames',Table(1,:))
        writetable(T, ['_resultados/' dataset '.csv'])
        clear Table features ranking;
        
    end

end






