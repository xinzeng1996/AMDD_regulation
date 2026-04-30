%% ------------------------------------------------------------------------
% Dynamic Mean Field (DMF) Model with Iterative SC Optimization
%
% Key features:
% 1. Initial global coupling (G) search across a predefined range
% 2. Subsequent iterations fix G and refine SC
% 3. SC is iteratively updated based on FC mismatch 
% 4. LEiDA metrics and KL distance guide optimization
%
% ------------------------------------------------------------------------

clear; close all;

addpath ../modeling/functions
rng('default');

%% -------------------- Load empirical data --------------------
load ../datas/empiricalLEiDA_BOLDS.mat
P1emp=mean(P1emp); P2emp=mean(P2emp); P3emp=mean(P3emp);
LT1emp=mean(LT1emp); LT2emp=mean(LT2emp); LT3emp=mean(LT3emp);

load '../datas/dep2_meanfc.mat';
load '../datas/dep2_meansc.mat';

%% -------------------- Preprocess SC & FC --------------------
SC = dep2_meansc(1:90,1:90);
SC = SC([1:2:90,90:-2:2],[1:2:90,90:-2:2]);
SC = (SC-min(SC(:))) / (max(SC(:))-min(SC(:)));

FC = dep2_meanfc(1:90,1:90);
FC = FC([1:2:90,90:-2:2],[1:2:90,90:-2:2]);
FC(1:size(FC,1)+1:end)=0;
FC = FC / max(FC(:));

FC_emp = FC;
N = size(SC,1);
NumClusters = size(Vemp,1);

%% -------------------- DMF parameters --------------------
G_coupling = 0.4:0.025:0.75;
Isubdiag = find(tril(ones(N),-1));

dt=0.1; tmax=340000;
tspan = 0:dt:tmax;

taon=100; taog=10;
gamma_factor=0.641; sigma_factor=0.01;

JN=0.15; I0=0.382;
Jexte=1; Jexti=0.7; w=1.4;

tcut=20*1000; ds=800;
nTrials_use=10;
SC1 = SC;
G_coupling2 = 0;
G_kld2 = 0;
Tolerance = 1;
ToleranceStep = 0.0125;
%% -------------------- Parallel --------------------
delete(gcp('nocreate'));
core_use = nTrials_use;
c = parpool(core_use);
c.IdleTimeout = 2400;
%% -------------------- Iterative Optimization --------------------
tic;
flag = 0; differ(1)=1; iT=1;
while iT <= 1000
    iT=iT+1;
    G_coupling1 = 0;  G_kld1 = 0;
    nTrials = nTrials_use;
    r_cc = zeros(length(G_coupling),1);

    % -------- G search strategy --------
    % First iteration: search across full G range
    % Later iterations: fix G using best index
    if flag==1
        G_coupling=G_coupling(G_coupling_best);
    end
    G_coupling_best=1;

    %% -------------------- G Loop --------------------
    for ii = 1 : length(G_coupling)
        cb1 = zeros(length(Isubdiag),nTrials);
        Cb1 = zeros(N,N,nTrials);
        G_coupling_local = G_coupling(ii);
        [~,J]=Balance_J5_opt(G_coupling_local,SC1,1);

        %% -------------------- Trial Simulation (Parallel) --------------------
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

            %% ---- Neural Dynamics ----
            for i = 1 : 1 : length(tspan)   

                xn=I0*Jexte+w*JN*sn+G_coupling_local*JN*SC1*sn-J.*sg;
                xg=I0*Jexti+JN*sn-sg;
                rn=phie(xn);
                rg=phii(xg);
                sn=sn+dt*(-sn/taon+(1-sn)*gamma_factor.*rn./1000.)+sqrt(dt)*sigma_factor*randn(N,1);
                sn(sn>1) = 1;
                sn(sn<0) = 0;
                sg=sg+dt*(-sg/taog+rg./1000.)+sqrt(dt)*sigma_factor*randn(N,1);
                sg(sg>1) = 1;
                sg(sg<0) = 0;


                rn(rn<0)=0;
                rg(rg<0)=0;
                rn(rn>100)=100;
                rg(rg>100)=100;
                rE=rn;rI=rg;


                if(rem(i,1/dt)==0)
                    TSE_downsampled(:,floor(i/(1/dt))) = rE;
                    TSI_downsampled(:,floor(i/(1/dt))) = rI;
                end

                j = j + 1;
                if j == (1/dt)
                    neuro_FR(nn,:) = rE';
                    neuro_FRi(nn,:) = rI';
                    nn = nn + 1;
                    j = 0;
                end

            end
            neuro_FR1(:,:,tr) = neuro_FR(:,:);
            neuro_FR1i(:,:,tr) = neuro_FRi(:,:);
            %% ---- BOLD Simulation ----
            BOLD_Signal=zeros(N,tmax);
            for i=1:N
                BOLD_Signal(i,:)=BOLD1(tmax/1000,TSE_downsampled(i,:)',1000)';
            end
            
            BOLD_Signal=BOLD_Signal(:,tcut + 1:ds:end);
            BOLDs(:,:,tr)=BOLD_Signal;

            %% ---- FC Computation ----
            Cb1(:,:,tr) = corrcoef(BOLD_Signal');
            Cb1(:,:,tr)=atanh(Cb1(:,:,tr));
            Cb1(:,:,tr)=Cb1(:,:,tr)-diag(diag(Cb1(:,:,tr)));
            Cb1_flag =Cb1(:,:,tr);
            Cb1_lowlable=Cb1_flag(Isubdiag);
            cb1(:,tr) = atanh(Cb1_lowlable);
        end

        %% -------------------- LEiDA & Metrics --------------------
        [PTRsim,Pstates,LTime]=LEiDA_fix_cluster2_zx_allTrial(BOLDs,NumClusters,Vemp,ds/1000);
        errorlifetime=sqrt(sum((LT3emp-LTime).^2)/length(LTime));
        tmp=0.5*(sum(Pstates.*log(Pstates./P3emp))+sum(P3emp.*log(P3emp./Pstates)));
        tmp(isnan(tmp))=0; kldist=tmp;
        entropydist=EntropyMarkov2(PTR3emp,PTRsim,P3emp,Pstates);
        allPstates=Pstates; allPTR=PTRsim; allLife=LTime;

        %% -------------------- FC Similarity --------------------
        Coef= sum(Cb1,3)/nTrials;
        for k=1:N
            Coef(k,k)=0;
        end
        kldist=mean(kldist);
        CB_col=(Coef(Isubdiag));
        FC_col=FC_emp(Isubdiag);
        r_c = corrcoef(CB_col,FC_col);
        r_cc(ii,1) = r_c(2);
        kldists(ii,1) = kldist;

        %% -------------------- Best G Selection --------------------
        if (1/kldist)>G_kld1
            G_coupling1 = r_c(2);
            G_kld1 = 1/kldist;
            kld_opt = kldist;
            G_Coef = Coef;
            G_coupling_best = ii;
            best_FR1=sum(neuro_FR1,3)/nTrials;
            best_FR1i=sum(neuro_FR1i,3)/nTrials;
            bolds = BOLDs;
            allPstates_opt = allPstates; allPTR_opt = allPTR; allLife_opt = allLife;
            allErrorlife_opt = errorlifetime; allEntropydist_opt = entropydist;
        end

    end

    if Tolerance == 1
        save('../datas/dep2/WC_G_Coef(initial)','G_Coef');
    end

    if Tolerance==1
        save('../datas/dep2/G_couplingT1','G_coupling');
        save('../datas/dep2/r_ccWC','r_cc');
        save('../datas/dep2/kldistT1','kldists');
    end

    disp(strcat('G_coupling:',num2str(G_coupling(G_coupling_best))));

    kld_opts(iT-1,1) = kld_opt;   % add distance
    kld_opts(iT-1,2) = Tolerance;

    Tolerance = Tolerance - ToleranceStep;

    %% -------------------- SC Update --------------------
    SC2 = SC1;
    load ../datas/signal_dep2_90p.mat
    edep2_bold = signal_dep2_90p;
    for i=1:size(edep2_bold,3)
        singlePCe(:,:,i)=calculatePC(edep2_bold(:,:,i));
        singlePRe(:,:,i)=calculatePR(edep2_bold(:,:,i));
        singlePRe(:,:,i)=atanh(singlePRe(:,:,i));
    end

    FC_emp_pc=(mean(singlePCe,3));  % phase coherence
    for i=1:size(bolds,3)
        singlePCs(:,:,i)=calculatePC(bolds(:,:,i));
        singlePRs(:,:,i)=calculatePR(bolds(:,:,i));
        singlePRs(:,:,i)=atanh(singlePRs(:,:,i));
    end

    G_Coef_pc=(mean(singlePCs,3));
    G_Coef_pr=(mean(singlePRs,3));
    G_Coef_pr=(G_Coef_pr)./max(max((G_Coef_pr)));

    for i=1:size(FC_emp_pc,1)
        FC_emp_pc(i,i)=0;
        G_Coef_pc(i,i)=0;
    end

    differ(iT) = norm(FC_emp_pc-G_Coef_pc,'fro'); prtmp = corrcoef(FC_emp,G_Coef_pr); prcorr(iT)=prtmp(2); diffMAX(iT)=max(max(abs(FC_emp_pc-G_Coef_pc)));
    figure(10);subplot(411);plot(iT,differ(iT),'rd');hold on;subplot(412);plot(iT,diffMAX(iT),'kp');hold on;subplot(413);plot(iT,prcorr(iT),'g*');hold on;
    subplot(414);plot(iT,kld_opts(iT-1,1),'b*'); hold on  % kld distance

    for k1=1:N
        for k2=1:N
            if abs(FC_emp(k1,k2)-G_Coef_pr(k1,k2))>0.025
                if FC_emp(k1,k2)>0
                    SC1(k1,k2) = SC1(k1,k2) + 0.002 .* (FC_emp_pc(k1,k2)-G_Coef_pc(k1,k2));
                else
                    if SC(k1,k2)==0
                        SC1(k1,k2)=0;
                    else
                        SC1(k1,k2)=0.0005;
                    end
                end
            end
        end
    end
    SC1(SC1<0)=0;

    flag = 1;

    if iT==500; G_kld2 = 0; end;
    if G_kld1 > G_kld2
        G_coupling2 = G_coupling1;
        G_kld2=G_kld1;
        B_best_FR1=best_FR1;
        B_best_FR1i=best_FR1i;

        if iT>500 % stable after 500 iterations
            now_opt_PC = G_Coef_pc; now_opt_PR = G_Coef_pr; now_opt_EC = SC2; all_eFC=singlePRe; all_sFC=singlePRs;
            allPstates_opt2=allPstates_opt; allPTR_opt2 = allPTR_opt; allLife_opt2 = allLife_opt;
            allErrorlife_opt2 = allErrorlife_opt; allEntropydist_opt2 = allEntropydist_opt;
            allbolds2 = bolds;
        end
    end

    disp(num2str(iT));
end

disp('Done');

%% -------------------- Save results --------------------
mean1FR1=mean(B_best_FR1,1);
mean2FR1=mean(B_best_FR1,2);
mean1FR1i=mean(B_best_FR1i,1);
mean2FR1i=mean(B_best_FR1i,2);

data_need_cluster = struct( ...
    'allPstates_opt2',allPstates_opt2, ...
    'allPTR_opt2',allPTR_opt2, ...
    'allLife_opt2',allLife_opt2, ...
    'allErrorlife_opt2',allErrorlife_opt2,...
    'allEntropydist_opt2',allEntropydist_opt2, ...
    'allbolds2',allbolds2, ...
    'kld_opts',kld_opts, ...
    'differ',differ, ...
    'diffMAX',diffMAX, ...
    'prcorr',prcorr, ...
    'SCemp',SC, ...
    'EC',now_opt_EC,...
    'FCemp',FC_emp, ...
    'FCsim',now_opt_PR, ...
    'all_eFC',all_eFC, ...
    'all_sFC',all_sFC, ...
    'phaseCoherenceSim',now_opt_PC, ...
    'phaseCoherenceEmp',FC_emp_pc, ...
    'mean1FR1',mean1FR1, ...
    'mean2FR1',mean2FR1, ...
    'mean1FR1i',mean1FR1i, ...
    'mean2FR1i',mean2FR1i);

save('../datas/dep2/data_need_cluster_finalDMF_testIteration_cb1atanh_allTrial','data_need_cluster')

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
