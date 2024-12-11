function [model,dataTable] = model_trf(eegPath,stimPath,fs,Dir,channel,tmin,tmax,lambda,norma,nfold,scale)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Default directories are as follow:
    % eegPath = 'E:/PhD/data/Di_Liberto/transformed/Natural Speech/EEG/'
    % stimPath = 'E:/PhD/data/Di_Liberto/transformed/Natural Speech/stimuli/value_cosine_bert.mat'
    % Normalization strategies: (norma)
    %       0: zscore
    %       1: min-max
    %       2: scaling only
    %       3: scaling + zscore
    %       others: skip normalization
    %
    % Scaling of stimulus: (scale)
    %       0: linear
    %       1: log 
    %       2: power law
    
    % load and reform the data

    % switch eeglab
    eeglab;
    stimValues = load(stimPath);
    % create empty containers
    eegContainer = [];
    stimContainer = [];
    % iterate folders of each run
    val_idx = 1;
    for r = 1:20
        runPath = strcat(eegPath, 'Run', string(r),'/interpolated/');
        %fprintf('Searching in path: %s \n', runPath);
        fileList = dir(fullfile(runPath,'*.set'));
        % number of files preserved for each run
        num_files = length(fileList); 
        fprintf('Number of subjects: %d \n',num_files);
   
        switch num_files
            case 0
            % skip empty folders
            otherwise
                % retrieve inital and end time
                duration = stimValues.duration(val_idx,:);
                st = duration(1);
                ed = duration(2);
                % load corresponding semantic dissimilarity values
                base_vals = cell2mat(stimValues.valid_values(val_idx));
                
                % load all eeg datasets of one run
                for s = 1:num_files
                    path = fileList(s).folder;
                    name = fileList(s).name;
                    EEG = pop_loadset(name,path);
                    %resample the data to 64Hz
                    EEG = pop_resample(EEG,64);
                    % append eeg dataset to the data pool
                    % select one channel and pop out invalid time intervals
                    % to avoid out of range
                    if length(EEG.data) < ed
                        fin = length(EEG.data);
                    else 
                        fin = ed;
                    end
                    fprintf('\n duration: %d - %d \n',st,fin);
                    % truncate the data
                    selected_eeg = EEG.data(channel,st:fin);
                    % z-score normalization
                    switch norma
                        case 0
                            selected_eeg = zscore(selected_eeg);
                        case 1
                            selected_eeg = min_max(selected_eeg);
                        case 2
                            selected_eeg = selected_eeg / 1.0e+06;
                        case 3
                            selected_eeg = zscore(selected_eeg / 1.0e+06);
                        otherwise
                        % without normalization
                    end
                    % update the container
                    eegContainer = [eegContainer,{selected_eeg}];
                    % append stimulus data
                    selected_stim = base_vals(st:fin);
                    % apply scaling
                    switch scale
                        case 0
                        % no transfromation needed for linear scaling
                        case 1
                            selected_stim = logscale(selected_stim);
                        case 2
                            selected_stim = power_law(selected_stim,0.4);
                        otherwise
                            error('Invalid index for scaling.');
                    end
                          
                    stimContainer = [stimContainer,{selected_stim}];

                end
 
                % update the value index
                val_idx = val_idx + 1;
        end

        % clear out eeglab data
        EEG = [];
        eeglab redraw;
        fprintf('Finished loading: Run %d . \n', r);
    end

    disp('Start training...');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Default settings:
    % Dir = 1; % direction of causality
    % tmin = 0; % minimum time lag (ms)
    % tmax = 500; % maximum time lag (ms)
    % lambda = 0.1; % regularization parameter
    % nfold = 8; % number of folds

    % Train and evaluate the model using cross validation. We leave out one
    % subset for test.

    colnames = {'Cz','Pz','Fz'};
    cellArray = cell(1,nfold);
    dataMatrix = zeros(nfold,3);

    for i = 1:length(nfold)
        testTrial = i;
        % separate the data into trainset and testset
        [strainSet,rtrainSet,stestSet,rtestSet] = trf_partition(stimContainer,eegContainer,nfold,testTrial);
        % training
        model = mTRFtrain(strainSet,rtrainSet,fs,Dir,tmin,tmax,lambda,'zeropad',0);

        % evaluate the model
        [pred,test] = mTRFpredict(stestSet,rtestSet,model,'zeropad',0);
        cellArray{i} = [mean(test.r)];
        fprintf('Completed iteration: %d / %d',i,nfold);
    end

    for n = 1:length(cellArray) 
        dataMatrix(n, :) = cellArray{i}; 
    end 

    dataTable = array2table(dataMatrix, 'VariableNames', colnames);
    writetable(dataTable,'E:/PhD/data/Di_Liberto/transformed/Natural Speech/statistics/powlaw_w2v_cosine.csv');
end