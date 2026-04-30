%% ------------------------------------------------------------------------
% Regulation Simulation using DMF Model
%
% This script evaluates stimulation effects by:
% 1. Applying region-specific stimulation 
% 2. Simulating whole-brain dynamics under fixed SC and G
% 3. Comparing regulated dynamics with HC
%
% NOTE:
% - Visualization code removed for clarity
% ------------------------------------------------------------------------

clear all; close all

addpath ../modeling/functions

%% -------------------- Load empirical data --------------------
load ../datas/empiricalLEiDA_BOLDS.mat
P1emp=mean(P1emp); P2emp=mean(P2emp); P3emp=mean(P3emp);
LT1emp=mean(LT1emp); LT2emp=mean(LT2emp); LT3emp=mean(LT3emp);
rng('default');

% HC FC
load('../datas/fcmeanhc.mat')
FC=fcmean(1:90,1:90);
FC=FC([1:2:90,90:-2:2],[1:2:90,90:-2:2]);
for i=1:size(FC,1); FC(i,i)=0; end
FC = FC/(max(max(FC)));

% Load optimized SC from previous DMF fitting
load('../datas/dep1/data_need_cluster_finalDMF_testIteration_cb1atanh_allTrial.mat')
SC=data_need_cluster.EC;

% Load optimal G
load('../datas/dep1/G_couplingT1.mat'); 
load('../datas/dep1/kldistT1.mat');
[~,gidx]=(min(kldists));
G_coupling = G_coupling(gidx);
[~,J0]=Balance_J5_opt(G_coupling,SC,1);

%% -------------------- Basic settings --------------------
FC_emp=FC;
N = size(SC,1);
NumClusters=size(Vemp,1);

%% -------------------- DMF parameters --------------------
Isubdiag = find(tril(ones(N),-1));
ds = 800;

dt = 0.1;
tmax = 340000;
tspan = 0 : dt : tmax;

taon=100; taog=10;
gamma_factor=0.641;
sigma_factor=0.01;

JN=0.15; I0=0.382;
Jexte=1.; Jexti=0.7; w=1.4;
tcut=20*1000;

nTrials_use = 1;
SC1 = SC;

%% -------------------- Parallel setup --------------------
delete(gcp('nocreate'));
core_use = 1;
c = parpool(core_use);
c.IdleTimeout = 2400;

%% -------------------- Stimulation settings --------------------
sti_strength = [-0.038:0.001:0.038];
sti_area = zeros(90,1);
sti_area([2,89,4,87,12,79])=1;   % target_regions parameter

%% -------------------- Iteration over stimulation --------------------
tic;

flag = 0; differ(1)=1;
G_kld1 = 0;

iT=1;
while iT <= length(sti_strength)

    %% -------- Apply stimulation --------
    me = 1 + sti_strength(iT).*sti_area;
    mi = 1;

    me(me==0)=1e-6; 
    mi(mi==0)=1e-6;

    G_coupling1 = 0;
    nTrials = nTrials_use;

    cb1 = zeros(length(Isubdiag),nTrials);
    Cb1 = zeros(N,N,nTrials);

    G_coupling_local = G_coupling;
    J=J0;

    %% -------------------- Simulation (parallel trials) --------------------
    parfor tr = 1 : nTrials
        
        rE=0.1*ones(N,1);
        rI=0.1*ones(N,1);
        sn=0.001*ones(N,1);
        sg=0.001*ones(N,1);

        data_len=tmax/dt/10;
        TSE_downsampled = 1.1*ones(N,data_len,'single');
        TSI_downsampled = 1.1*ones(N,data_len,'single');

        neuro_FR = zeros(tmax,N);
        neuro_FRi = zeros(tmax,N);
        nn = 1; j = 0;

        %% ---- Neural dynamics ----
        for i = 1 : length(tspan)

            xn=I0*Jexte+w*JN*sn+G_coupling_local*JN*SC1*sn-J.*sg;
            xg=I0*Jexti+JN*sn-sg;

            rn=phie_fic_test(xn,me);
            rg=phii_fic(xg,mi);

            sn=sn+dt*(-sn/taon+(1-sn)*gamma_factor.*rn./1000.)+sqrt(dt)*sigma_factor*randn(N,1);
            sn(sn>1)=1; sn(sn<0)=0;

            sg=sg+dt*(-sg/taog+rg./1000.)+sqrt(dt)*sigma_factor*randn(N,1);
            sg(sg>1)=1; sg(sg<0)=0;

            rn(rn<0)=0; rn(rn>100)=100;
            rg(rg<0)=0; rg(rg>100)=100;

            rE=rn; rI=rg;

            if(rem(i,1/dt)==0)
                TSE_downsampled(:,floor(i/(1/dt))) = rE;
                TSI_downsampled(:,floor(i/(1/dt))) = rI;
            end

            j=j+1;
            if j==(1/dt)
                neuro_FR(nn,:) = rE';
                neuro_FRi(nn,:) = rI';
                nn=nn+1; j=0;
            end
        end

        neuro_FR1(:,:,tr)=neuro_FR;
        neuro_FR1i(:,:,tr)=neuro_FRi;

        %% ---- BOLD ----
        BOLD_Signal=zeros(N,tmax);
        for i=1:N
            BOLD_Signal(i,:)=BOLD1(tmax/1000,TSE_downsampled(i,:)',1000)';
        end
        BOLD_Signal=BOLD_Signal(:,tcut + 1:ds:end);
        BOLDs(:,:,tr)=BOLD_Signal;

        %% ---- FC ----
        Cb1(:,:,tr) = corrcoef(BOLD_Signal');
        Cb1(:,:,tr)=Cb1(:,:,tr)-diag(diag(Cb1(:,:,tr)));
        Cb1_flag =Cb1(:,:,tr);
        Cb1_lowlable=Cb1_flag(Isubdiag);
        cb1(:,tr) = atanh(Cb1_lowlable);
    end

    %% -------------------- LEiDA & evaluation --------------------
    [PTRsim,Pstates,LTime]=LEiDA_fix_cluster2_zx_allTrial(BOLDs,NumClusters,Vemp,ds/1000);
    errorlifetime=sqrt(sum((LT1emp-LTime).^2)/length(LTime));
    tmp=0.5*(sum(Pstates.*log(Pstates./P1emp))+sum(P1emp.*log(P1emp./Pstates)));
    tmp(isnan(tmp))=0; kldist=mean(tmp);
    entropydist=EntropyMarkov2(PTR1emp,PTRsim,P1emp,Pstates);
    allPstates=Pstates; allPTR=PTRsim; allLife=LTime;

    Coef= sum(Cb1,3)/nTrials;
    for k=1:N
        Coef(k,k)=0;
    end
    
    kldist=mean(kldist);
    CB_col=atanh(Coef(Isubdiag));
    FC_col=FC_emp(Isubdiag);
    r_c = corrcoef(CB_col,FC_col);
    kldistsFIC(iT) = kldist;

    Coef = atanh(Coef); 
    Coef = Coef./max(max(Coef));
    Cb1_all(iT,:,:,:) = Cb1;

    gbcComp = mean(abs( mean(Coef)-mean(FC_emp) ));
    if (1/gbcComp)>G_kld1 
        G_coupling1 = r_c(2);
        G_kld1 = (1/gbcComp);
        kld_opt = kldist;
        G_Coef = Coef;
        best_FR1=sum(neuro_FR1,3)/nTrials;
        best_FR1i=sum(neuro_FR1i,3)/nTrials;
        bestbolds = BOLDs;
        allPstates_opt = allPstates; allPTR_opt = allPTR; allLife_opt = allLife;
        allErrorlife_opt = errorlifetime; allEntropydist_opt = entropydist;
        iTindex = iT;
    end

    %% -------------------- HC reference comparison --------------------
    load ../datas/signal_hc_90p.mat
    ehc_bold = signal_hc_90p;
    for i=1:size(ehc_bold,3)
        singlePCe(:,:,i)=calculatePC(ehc_bold(:,:,i));
        singlePRe(:,:,i)=calculatePR(ehc_bold(:,:,i));
    end

    FC_emp_pc=(mean(singlePCe,3));  % phase coherence
    FC_emp_pr=(mean(singlePRe,3));  % pearson correlation
    FC_emp_pr=atanh(FC_emp_pr)./max(max(atanh(FC_emp_pr)));

    for i=1:size(BOLDs,3)
        singlePCs(:,:,i)=calculatePC(BOLDs(:,:,i));
        singlePRs(:,:,i)=calculatePR(BOLDs(:,:,i));
    end

    G_Coef_pc=(mean(singlePCs,3));
    G_Coef_pr=(mean(singlePRs,3));
    G_Coef_pr=atanh(G_Coef_pr)./max(max(atanh(G_Coef_pr)));

    for i=1:size(FC_emp_pc,1)
        FC_emp_pc(i,i)=0;
        G_Coef_pc(i,i)=0;
    end

    differ(iT) = norm(FC_emp_pc-G_Coef_pc,'fro'); 
    prtmp = corrcoef(FC_emp_pr,G_Coef_pr); prcorr(iT)=prtmp(2); 
    diffMAX(iT)=max(max(abs(FC_emp_pc-G_Coef_pc)));
    GBCdiffCoef(iT) =  mean(abs( mean(Coef)-mean(FC_emp) ));
    Allstates(:,iT) = allPstates; 
    AllCoef(:,:,iT) = Coef;

    flag = 1;
    disp(num2str(iT));
    iT=iT+1;
end

disp('Done');

%% -------------------- Save results --------------------
mean1FR1=mean(best_FR1,1);
mean2FR1=mean(best_FR1,2);
mean1FR1i=mean(best_FR1i,1);
mean2FR1i=mean(best_FR1i,2);

data_need_cluster = struct( ...
    'Allstates',Allstates, ...
    'GBCdiffCoef',GBCdiffCoef, ...
    'AllCoef',AllCoef, ...
    'iTindex',iTindex, ...
    'allPstates_opt',allPstates_opt, ...
    'allPTR_opt',allPTR_opt, ...
    'allLife_opt',allLife_opt, ...
    'allErrorlife_opt',allErrorlife_opt,...
    'allEntropydist_opt',allEntropydist_opt, ...
    'allbolds',bestbolds, ...
    'kldistsFIC',kldistsFIC, ...
    'differ',differ, ...
    'diffMAX',diffMAX, ...
    'prcorr',prcorr,...
    'FCsim',G_Coef, ...
    'phaseCoherenceEmp',FC_emp_pc, ...
    'mean1FR1',mean1FR1, ...
    'mean2FR1',mean2FR1, ...
    'mean1FR1i',mean1FR1i, ...
    'mean2FR1i',mean2FR1i, ...
    'Cb1_all',Cb1_all);

save('./dataSave/data_need_ficdep1_dlpfc_100','data_need_cluster')

toc




%% -------------------- Utility functions --------------------
function result=calculatePR(data)
a=corrcoef(data');
for i=1:size(a,1)
    a(i,i)=0;
end
result=a;
end

function result=calculatePC(data)

addpath('../modeling/functions')
TR=0.8;
n_Subjects=1;
[N_areas, Tmax]=size(data);
% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.04;                    % lowpass frequency of filter (Hz)
fhi = 0.09;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter
clear fnq flp fhi Wn k


for s=1:n_Subjects
    [N_areas, Tmax]=size(data);
    % Get the BOLD signals from this subject in this condition
    BOLD = data;
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
        iFC_all(:,:,t) = iFC;
    end
    result=mean(iFC_all,3);
end

end
