%
close all; clear all; clc;
IsFrist= 0;
%% load data:
mainfilename = 'E:\Stroke\';
folder_names = {'F','M','FC','MC'};
session_type = {'wav_arrayMic','wav_headMic'};
ii = 1;i_p=1;
if (IsFrist)
for f = 1 : length(folder_names)
    subject_name = dir([mainfilename,folder_names{f},'\',folder_names{f}]);
    path_name = subject_name(1).folder;
    subject_name = {subject_name(3:end).name};
    for s = 1 : length(subject_name)
        session_names = dir([path_name,'\',subject_name{s}]);
        path_name_temp = session_names(1).folder;
        session_names = {session_names(3+1:end).name};
        for se = 1 : length(session_names)
            data_names = dir([path_name_temp,'\',session_names{se},'\',session_type{1}]);
            if isempty(data_names)
                data_names = dir([path_name_temp,'\',session_names{se},'\',session_type{2}]);
            end
            data_path_temp = data_names(1).folder;
            data_names = {data_names(3:end).name};
            for d = 1 : length(data_names)
                filename = [data_path_temp,'\',data_names{d}];
                [all_data{ii},Fs(f,s,se,d)] = audioread(filename);
                if Fs(f,s,se,d) == 0
                    disp([folder_names{f},subject_name{s},session_names{se},data_names{d}])
                end
                all_label{ii} = folder_names{f};
                ii = ii + 1;
            end
        end
    end
    labels_temp = all_label(i_p:ii-1);
    data_temp = all_data(i_p:ii-1);
    load_data_temp.data_temp = data_temp;
    load_data_temp.labels_temp = labels_temp;
    save(folder_names{f},'load_data_temp')
    i_p = ii;
end
else
    all_data=[];all_label=[];
    for f = 1 : length(folder_names)
        load(folder_names{f});
        all_data = [all_data,load_data_temp.data_temp];
        all_label = [all_label,load_data_temp.labels_temp];
    end
end
    
labels_name = folder_names;    
%save('data_dist', 'all_data','all_label')
% sound(y,Fs)
%% data visualization:
figure()
h=histogram(categorical(all_label));

figure()
for f = 1 : length(labels_name)
    data_temp = all_data(strcmp(all_label,labels_name{f}));
    len_temp = cellfun(@numel, data_temp);
    histogram(len_temp);
    hold on
end
legend(labels_name)

%% cutting data and spliting:

min_count = min(h.Values);
ratio.train = 0.7; ratio.val = 0.0; ratio.test = 0.3;
ratio_type = {'train','val','test'};
X.train=[]; Xlabel.train=[]; Y.train=[];
X.val=[]; Xlabel.val=[]; Y.val=[];
X.test=[]; Xlabel.test=[]; Y.test=[];
for f = 1 : length(labels_name)
    data_temp = all_data(strcmp(all_label,labels_name{f}));
    rand_ind = randperm(numel(data_temp),min_count);
    if contains(labels_name{f},'C')
        la = 0;
    else
        la = 1;
    end
    ind_split_f=0;
    for k = 1 :length(ratio_type)
        ind_split_i = ind_split_f+1;
        ind_split_f = ind_split_i-1+floor(min_count*ratio.(ratio_type{k}));
        ind_temp = rand_ind(ind_split_i:ind_split_f);
        X.(ratio_type{k}) = [X.(ratio_type{k}), data_temp(ind_temp)];
        Xlabel.(ratio_type{k}) = [Xlabel.(ratio_type{k}),...
            repmat(labels_name(f),1,ind_split_f-ind_split_i+1)];
        Y.(ratio_type{k}) = [Y.(ratio_type{k}),...
            repmat(la,1,ind_split_f-ind_split_i+1)];
    end
end

%% sort:
for k = 1 :length(ratio_type)
    
    numObservations = numel(X.(ratio_type{k}));
    sequenceLengths = cellfun(@numel, X.(ratio_type{k}));
    [sequenceLengths,idx] = sort(sequenceLengths);
    X.([ratio_type{k},'_sort']) = X.(ratio_type{k})(idx).';
    Xlabel.([ratio_type{k},'_sort'])=Xlabel.(ratio_type{k})(idx).';
    Y.([ratio_type{k},'_sort']) = Y.(ratio_type{k})(idx).';
    
    %plot
    figure
    bar(sequenceLengths)
    xlabel("Sequence")
    ylabel("Length")
    title("Sorted Data")
end

miniBatchSize = 2^5; %32

%% net

inputSize = 1;
numHiddenUnits = 20;
numClasses = 2;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 100;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
%    'ValidationData',[X.val_sort,Y.val_sort],...
%%
XTrain = cellfun(@(c) c.',X.train_sort,'UniformOutput' ,false);
YTrain = categorical(Y.train_sort);
net = trainNetwork(XTrain,YTrain,layers,options);


save('net_lstm_bd','net')

%% train
