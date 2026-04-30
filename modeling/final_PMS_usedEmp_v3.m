
%% modified from demo/pnas-neuromod-master/LEiDA_PsiloData.m in Kringelbach et al., PNAS, 2020.
%%


close all; clear all;
flaggg=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Dynamic coupling of Whole-Brain Neuronal and Neurotransmitter Systems
%     Kringelbach, M. L., Cruzat, J., Cabral, J., Knudsen, G. M.,
%       Carhart-Harris, R. L., Whybrow, P. C., Logothetis N. K. & Deco, G.
%         (2020) Proceedings of the National Academy of Sciences

%   Barcelona?Spain, March, 2020.

%%%%%%

addpath('/your_data_folder/demo/pnas-neuromod-master/Functions')  % from Kringelbach et al., PNAS, 2020.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1 - Compute the Leading Eigenvectors from the BOLD datasets
disp('Processing the eigenvectors from BOLD data')
% Load here the BOLD data (which may be in different formats)
% Here the BOLD time courses in AAL parcellation are organized as cells,
% where tc_aal{1,1} corresponds to the BOLD data from subject 1 in
% condition 1 and contains a matrix with lines=N_areas and columns=Tmax.

load /your_data_folder/dataHC/signal_hc_90p.mat
ehc_bold = signal_hc_90p;
load /your_data_folder/dataDep/1_signal/signal_dep1_90p.mat
edep1_bold = signal_dep1_90p;
load /your_data_folder/dataDep/1_signal/signal_dep2_90p.mat
edep2_bold = signal_dep2_90p;

cellArray = cell(63, 3);

for i = 1:63
    cellArray{i, 1} = ehc_bold(:, :, i);
end
for i = 1:35
    cellArray{i, 2} = edep1_bold(:, :, i);
end
for i = 1:29
    cellArray{i, 3} = edep2_bold(:, :, i);
end
data=cellArray;

[n_Subjects, n_cond]=size(data);
[N_areas, Tmax]=size(data{1,1});

% Parameters of the data
TR=0.8;  % Repetition Time (seconds)

% Preallocate variables to save FC patterns and associated information
Leading_Eig=zeros(Tmax*n_Subjects,1*N_areas); % All leading eigenvectors
Time_all=zeros(2, n_Subjects*Tmax); % vector with subject nr and cond at each t
t_all=0; % Index of time (starts at 0 and will be updated until n_Sub*Tmax)

% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.04;                    % lowpass frequency of filter (Hz)
fhi = 0.09;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency

k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter
clear fnq flp fhi Wn k

for s=1:n_Subjects
    for cond=1:n_cond
        if cond==2;n_Subjects=35; elseif cond==3;n_Subjects=29;end;
        [N_areas, Tmax]=size(data{s,cond});
        % Get the BOLD signals from this subject in this condition
        BOLD = data{s,cond};
        %BOLD = normalize(BOLD,'zscore');
        Phase_BOLD=zeros(N_areas,Tmax);

        % Get the BOLD phase using the Hilbert transform
        for seed=1:N_areas
            ts=demean(detrend(BOLD(seed,:)));
            signal_filt =filtfilt(bfilt,afilt,ts);
            Phase_BOLD(seed,:) = angle(hilbert(signal_filt));
        end

        for t=1:Tmax

            %Calculate the Instantaneous FC (BOLD Phase Synchrony)
            iFC=zeros(N_areas);
            for n=1:N_areas
                for p=1:N_areas
                    iFC(n,p)=cos(Phase_BOLD(n,t)-Phase_BOLD(p,t));
                end
            end

            % Get the leading eigenvector

            [V1,~]=eigs(iFC,1);
            % Make sure the largest component is negative
            if mean(V1>0)>.5
                V1=-V1;
            elseif mean(V1>0)==.5 && sum(V1(V1>0))>-sum(V1(V1<0))
                V1=-V1;
            end
            %%
            if t==50 && flaggg==1; figure;set(gcf,"Position",[700,500,230,200]);imagesc(iFC);colormap("hot"); set(gca,"Visible","off");
                figure;set(gcf,"Position",[700,500,150,60]);
                for iv1=1:90;if V1(iv1)>0;bar(iv1,V1(iv1),0.4,'FaceColor','r','EdgeColor','r');hold on;
                else; bar(iv1,V1(iv1),0.5,'FaceColor','k','EdgeColor','k'); hold on; end;end; xticks('');xlabel('Brain area')
            set(gcf,"PaperPositionMode",'auto'); set(gca,"FontName",'Arial','FontSize',9);box off
            %title('Leading eigenvector(t=50)')
            figure;set(gcf,"Position",[700,500,120,180]); for iv1 = 1:90;if V1(iv1) > 0;barh(iv1, V1(iv1), 0.4, 'FaceColor', 'r', 'EdgeColor', 'r');
                    hold on; else barh(iv1, V1(iv1), 0.5, 'FaceColor', 'k', 'EdgeColor', 'k');hold on;end;end
            yticks(''); ylabel('Brain area'); set(gcf,"PaperPositionMode",'auto');set(gca,"FontName",'Arial','FontSize',9);box off;
            flaggg=2;
            end
            %%
            % Save V1 from all frames in all fMRI sessions in Leading eig
            t_all=t_all+1; % Update time
            Leading_Eig(t_all,:)=V1; %vertcat(V1,V2);
            Time_all(:,t_all)=[s cond]; % Information that at t_all, V1 corresponds to subject s in a given condition
        end
    end
end
clear BOLD tc_aal signal_filt iFC VV V1 V2 Phase_BOLD

%% 2 - Cluster the Leading Eigenvectors
[n_Subjects, n_cond]=size(data);
disp('Clustering the eigenvectors into')
% Leading_Eig is a matrix containing all the eigenvectors:
% Collumns: N_areas are brain areas (variables)
% Rows: Tmax*n_Subjects are all time points (independent observations)

% Set maximum/minimum number of clusters
% There is no fixed number of states the brain can display
% Extending depending on the hypothesis of each work
mink=2;
maxk=4;
rangeK=mink:maxk;
flag=0;

% Set the parameters for Kmeans clustering
while flag == 0
    Kmeans_results=cell(size(rangeK));
    for k=1:length(rangeK)
        disp(['- ' num2str(rangeK(k)) ' clusters'])
        [IDX, C, SUMD, D]=kmeans(Leading_Eig,rangeK(k),'Distance','sqeuclidean','Replicates',20,'Display','off');
        Kmeans_results{k}.IDX=IDX;   % Cluster time course - numeric collumn vectos
        Kmeans_results{k}.C=C;       % Cluster centroids (FC patterns)
        Kmeans_results{k}.SUMD=SUMD; % Within-cluster sums of point-to-centroid distances
        Kmeans_results{k}.D=D;       % Distance from each point to every centroid
        ss=silhouette(Leading_Eig,IDX,'sqeuclidean');
        sil(k)=mean(ss);
    end

    distM_fcd=squareform(pdist(Leading_Eig,'euclidean'));
    dunn_score=zeros(length(rangeK),1);
    for j=1:length(rangeK)
        dunn_score(j)=dunns(rangeK(j),distM_fcd,Kmeans_results{j}.IDX);
        disp(['Performance for ' num2str(rangeK(j)) ' clusters'])
    end

    if dunn_score(2)>0.014; flag=1;end; % determine the best clustering is steady every time
end

[~,ind_max]=max(dunn_score);

disp(['Best clustering solution: ' num2str(rangeK(ind_max)) ' clusters']);

%% 3 - Analyse the Clustering results

% For every fMRI scan calculate probability and lifetimes of each pattern c.
P=zeros(n_cond,n_Subjects,maxk-mink+1,maxk);
LT=zeros(n_cond,n_Subjects,maxk-mink+1,maxk);

for k=1:length(rangeK)
    for cond=1:n_cond
        if cond==2;n_Subjects=35; elseif cond==3;n_Subjects=29;else;n_Subjects=63; end;
        for s=1:n_Subjects

            % Select the time points representing this subject and task
            T=((Time_all(1,:)==s)+(Time_all(2,:)==cond))>1;
            Ctime=Kmeans_results{k}.IDX(T);

            for c=1:rangeK(k)
                % Probability
                P(cond,s,k,c)=mean(Ctime==c);

                % Mean Lifetime
                Ctime_bin=Ctime==c;

                % Detect switches in and out of this state
                a=find(diff(Ctime_bin)==1);
                b=find(diff(Ctime_bin)==-1);

                % We discard the cases where state sarts or ends ON
                if length(b)>length(a)
                    b(1)=[];
                elseif length(a)>length(b)
                    a(end)=[];
                elseif  ~isempty(a) && ~isempty(b) && a(1)>b(1)
                    b(1)=[];
                    a(end)=[];
                end
                if ~isempty(a) && ~isempty(b)
                    C_Durations=b-a;
                else
                    C_Durations=0;
                end
                LT(cond,s,k,c)=mean(C_Durations)*TR;
            end
        end
    end
end
[n_Subjects, n_cond]=size(data);

P_pval=zeros(maxk-mink+1,maxk);
LT_pval=zeros(maxk-mink+1,maxk);

disp('Test significance between Placebo and LSD')
for k=1:length(rangeK)
    disp(['Now running for ' num2str(k) ' clusters'])
    for c=1:rangeK(k)
        % Compare Probabilities
        a=squeeze(P(1,:,k,c));  % Vector containing Prob of c in Baselineline
        b=squeeze(P(2,:,k,c));  % Vector containing Prob of c in LSD
        stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],1000,0.05,'ttest');
        P_pval(k,c)=min(stats.pvals);

        % Comapre Lifetimes
        a=squeeze(LT(1,:,k,c));  % Vector containing Prob of c in Baselineline
        b=squeeze(LT(2,:,k,c));  % Vector containing Prob of c in LSD
        stats=permutation_htest2_np([a,b],[ones(1,numel(a)) 2*ones(1,numel(b))],1000,0.05,'ttest');
        LT_pval(k,c)=min(stats.pvals);
    end
end
disp('%%%%% LEiDA SUCCESSFULLY COMPLETED %%%%%%%')

%% 4 - Plot FC patterns and stastistics between groups

disp(' ')
disp('%%% PLOTS %%%%')
disp(['Choose number of clusters between ' num2str(rangeK(1)) ' and ' num2str(rangeK(end)) ])

for k=1:length(rangeK)
    Pmin_pval(k)=min(P_pval(k,1:rangeK(k)));
end

[pmin_pval k]=min(Pmin_pval);
disp(['Note: The minimal K with significant difference is detected with K=' num2str(rangeK(k)) ' (p=' num2str(pmin_pval) ')'])

numsig=-1;
for k=1:length(rangeK)
    nums=length(find(P_pval(k,1:rangeK(k))<0.05));
    if nums>numsig
        numsig=nums;
    end
end
disp(['Note: The K with max number of significant difference is detected with K=' num2str(rangeK(k)) ' (n=' num2str(numsig) ')'])

disp(['Note: Silohuette optimum:'])
sil

%%% Graphics

K = input('Number of clusters: ');
Number_Clusters=K;
Best_Clusters=Kmeans_results{rangeK==K};
k=find(rangeK==K);

%% plot clusters
C = Best_Clusters.C;
idx = Best_Clusters.IDX;
IDX =idx;
data = Leading_Eig;
%data = tsne(data, [], no_dims, initial_dims, perplexity);
mappedX = tsne(data, 'NumDimensions', 2, 'NumPCAComponents', size(data, 2), 'Perplexity', 30);

figure;
colors = [120 213 182; 255 165 167; 173 188 221; 178 189 219;178 189 219]/255;
colors = [255 0 0; 255 255 0; 0 255 0; 178 189 219;178 189 219]/255;

for i = 1:3
    scatter(mappedX(idx==i,1), mappedX(idx==i,2), 5 , 'filled', 'MarkerFaceColor' ,colors(i,:));
    hold on;
end

hold on;
scatter(mean(mappedX(idx==1,1)), mean(mappedX(idx==1,2)), 350, 'kx', 'LineWidth',8);
scatter(mean(mappedX(idx==2,1)), mean(mappedX(idx==2,2)), 350, 'kx', 'LineWidth',8);
scatter(mean(mappedX(idx==3,1)), mean(mappedX(idx==3,2)), 350, 'kx', 'LineWidth',8);
scatter(mean(mappedX(idx==4,1)), mean(mappedX(idx==4,2)), 350, 'kx', 'LineWidth',8);
hold off;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids');
title('K-Means Clustering Scatters-tSNE');


coeff = pca(data);
reduced_data = data * coeff(:, 1:90); a=1;b=2;
reduced_C = C * coeff(:, 1:90);
cluster_labels = IDX;

cluster1_data = reduced_data(cluster_labels == 1, :);
cluster2_data = reduced_data(cluster_labels == 2, :);
cluster3_data = reduced_data(cluster_labels == 3, :);

figure;
scatter(cluster1_data(:, a), cluster1_data(:, b), 2,'r','filled');
hold on;
scatter(cluster2_data(:, a), cluster2_data(:, b), 2,'y','filled');
hold on;
scatter(cluster3_data(:, a), cluster3_data(:, b), 2,'g','filled');
hold on;
scatter(reduced_C(:, a), reduced_C(:, b), 250, 'kx', 'filled', 'MarkerEdgeColor', 'k','LineWidth',5); legend('Cluster 1','Cluster 2','Cluster 3','Centers');
hold off;
title('K-Means Clustering Scatters-PCA');

%%
% Clusters are sorted according to their probability of occurrence
ProbC=zeros(1,K);
for c=1:K
    ProbC(c)=mean(Best_Clusters.IDX==c);
end
[~, ind_sort]=sort(ProbC,'descend');

% Get the K patterns
V=Best_Clusters.C(ind_sort,:);
[~, N]=size(Best_Clusters.C);
% Order=[1:2:N N:-2:2];

Vemp=V;
P1emp=squeeze(P(1,:,k,ind_sort));
P2emp=squeeze(P(2,:,k,ind_sort));P2emp = P2emp(1:35,:);
P3emp=squeeze(P(3,:,k,ind_sort)); P3emp = P3emp(1:29,:);
LT1emp=squeeze(LT(1,:,k,ind_sort));
LT2emp=squeeze(LT(2,:,k,ind_sort)); LT2emp = LT2emp(1:35,:);
LT3emp=squeeze(LT(3,:,k,ind_sort)); LT3emp = LT3emp(1:29,:);

meanp1emp=squeeze(mean(P1emp));
meanp2emp=squeeze(mean(P2emp));
meanp3emp=squeeze(mean(P3emp));
[~, ind_sort1]=sort(meanp1emp,'descend');
[~, ind_sort2]=sort(meanp2emp,'descend');
[~, ind_sort3]=sort(meanp3emp,'descend');


PTR1emp=zeros(K);
PTR2emp=zeros(K);
PTR3emp=zeros(K);
n_sub1=zeros(K,1);
n_sub2=zeros(K,1);
n_sub3=zeros(K,1);
for psylo=1:3
    for s=1:n_Subjects
        % Select the time points representing this subject and LSD
        T=((Time_all(1,:)==s)+(Time_all(2,:)==psylo))>1;
        Ctime=Kmeans_results{k}.IDX(T);

        if psylo==1
            i=1;
            for c1=ind_sort1
                j=1;
                for c2=ind_sort1
                    sumatr=0;
                    for t=1:length(Ctime)-1
                        if Ctime(t)==c1 && Ctime(t+1)==c2
                            sumatr=sumatr+1;
                        end
                    end
                    if length(find(Ctime(1:length(Ctime)-1)==c1)) ~= 0
                        PTR1emp(i,j)=PTR1emp(i,j)+sumatr/(length(find(Ctime(1:length(Ctime)-1)==c1)));
                    end
                    j=j+1;
                end
                if length(find(Ctime(1:length(Ctime)-1)==c1)) ~=0
                    n_sub1(i)=n_sub1(i)+1;
                end
                i=i+1;
            end
        end

        if psylo==2
            i=1;
            for c1=ind_sort2
                j=1;
                for c2=ind_sort2
                    sumatr=0;
                    for t=1:length(Ctime)-1
                        if Ctime(t)==c1 && Ctime(t+1)==c2
                            sumatr=sumatr+1;
                        end
                    end
                    if length(find(Ctime(1:length(Ctime)-1)==c1)) ~=0
                        PTR2emp(i,j)=PTR2emp(i,j)+sumatr/(length(find(Ctime(1:length(Ctime)-1)==c1)));
                    end
                    j=j+1;
                end
                if length(find(Ctime(1:length(Ctime)-1)==c1)) ~=0
                    n_sub2(i)=n_sub2(i)+1;
                end

                i=i+1;
            end
        end

        if psylo==3
            i=1;
            for c1=ind_sort3
                j=1;
                for c2=ind_sort3
                    sumatr=0;
                    for t=1:length(Ctime)-1
                        if Ctime(t)==c1 && Ctime(t+1)==c2
                            sumatr=sumatr+1;
                        end
                    end
                    if length(find(Ctime(1:length(Ctime)-1)==c1)) ~=0
                        PTR3emp(i,j)=PTR3emp(i,j)+sumatr/(length(find(Ctime(1:length(Ctime)-1)==c1)));
                    end
                    j=j+1;
                end
                if length(find(Ctime(1:length(Ctime)-1)==c1)) ~=0
                    n_sub3(i)=n_sub3(i)+1;
                end

                i=i+1;
            end
        end

    end
end
for i=1:K
    PTR1emp(i,:)=PTR1emp(i,:)/n_sub1(i);
    PTR2emp(i,:)=PTR2emp(i,:)/n_sub2(i);
    PTR3emp(i,:)=PTR3emp(i,:)/n_sub3(i);
end

save empiricalLEiDA_BOLDS.mat; %  Save mean group FC value of dep1 as ../datas/empiricalLEiDA_BOLDS.mat;



