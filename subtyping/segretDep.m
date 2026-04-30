%% ------------------------------------------------------------------------
% Subtype Clustering Based on Functional Connectivity (FC)
%
% This script performs subject-level clustering using FC features:
% 1. Extracts upper triangular FC features from each subject
% 2. Performs k-means clustering (k=2, determined via silhouette)
% 3. Separates subjects into subtype groups
%
% -------------------------------------------------------------------------

clear; close all;

%% -------------------- Load data --------------------
data_dir = 'your data folder';
files = dir(data_dir);
file_names = {files.name};

fc_vector = [];           % feature matrix (subjects × edges)
FClobe_group = [];        % FC matrices for each subject

%% -------------------- Feature extraction --------------------
for i = 1:length(file_names)-2
    load(fullfile(data_dir, file_names{i+2}));   % load fc & fcmean
    
    % Reorder FC (hemispheric interleaving as in original code)
    FC = fc(1:90,1:90);
    FC = FC([1:2:90, 90:-2:2], [1:2:90, 90:-2:2]);
    
    % Remove self-connections
    FC(1:size(FC,1)+1:end) = 0;
    
    FClobe = FC;
    FClobe_group(:,:,i) = FClobe;
    
    % Vectorize upper triangle
    FC_upper = triu(FClobe);
    fc_vec = nonzeros(FC_upper(:));
    
    fc_vector = [fc_vector; fc_vec'];   % subjects × features
end

%% -------------------- Determine optimal k --------------------
silhouetteValue = zeros(1,10);

for k = 2:10
    IDX_tmp = kmeans(fc_vector, k, ...
        'Distance','cityblock', ...
        'Replicates',20, ...
        'Display','final');
    
    silhouetteValue(k) = mean(silhouette(fc_vector, IDX_tmp));
end

% Optimal cluster number (pre-determined)
k = 2;

%% -------------------- Final clustering --------------------
[IDX, C] = kmeans(fc_vector, k, ...
    'Distance','cityblock', ...
    'Replicates',20, ...
    'Display','final');

%% -------------------- Separate subtype FC --------------------
FC_styp1 = [];
FC_styp2 = [];

ii = 1; jj = 1;

for i = 1:length(IDX)
    if IDX(i) == 1
        FC_styp1(:,:,ii) = FClobe_group(:,:,i);
        ii = ii + 1;
    elseif IDX(i) == 2
        FC_styp2(:,:,jj) = FClobe_group(:,:,i);
        jj = jj + 1;
    end
end

%% %%%%%%%%%%%%%%%%%%%%%
% Save mean group FC value of dep1 as ../datas/dep1_meanfc.mat;
% Save mean group FC value of dep2 as ../datas/dep2_meanfc.mat;
% Save mean group SC value of dep1 as ../datas/dep1_meansc.mat;
% Save mean group SC value of dep2 as ../datas/dep2_meansc.mat;

