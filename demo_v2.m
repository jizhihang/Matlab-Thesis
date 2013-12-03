%% Clear everything for sanity check

clear all; close all;
tStart = tic;

% load data
fprintf('\nLoading mat features.\n');

 load('featuresAllYPHogRduced+-0.5.mat','XhogR');
 load('allLabelsReduced+-0.5.mat','Yd');
  load('featuresFilenames.mat','Xfilenames');
%   
%   
%   fprintf('\nRun PCA on data.\n');
% [ePca, mPca, lambda] = cvPca(XallUnitNorm,2048);
% [XallPca ] = cvPcaProj(XallUnitNorm, ePca, mPca);
%load('featuresAllUnitNormalizedPca2048.mat','XallPca');
%   [Xallnorm,mu,sigma]= featureNormalize(Xd);
% %   
%   [ePca, mPca, lambda] = cvPca(Xallnorm,2048);
%  [XallPca ] = cvPcaProj(Xallnorm, ePca, mPca);
  
%  d1 = find(Xall(1,:) <=0.5& Xall(1,:)>=-0.5);
% Xd1=Xall(:,d1);
% Yd1=Yall(d1);
% d2 = find(Xd1(2,:) <=0.5& Xd1(2,:)>=-0.5);
% Xd=Xd1(:,d2);
% Yd=Yd1(d2);

%get 10 faces per person for test as in pubfig83 closed universe
[X,y,Xtest,Ytest,LabelsTest,Labels]= dataSplitTrainTest(XhogR,Yd,10,Xfilenames);
% Randomize and split the data X to training and validation set
perTrain=0.85;
%Mean normalization of the Hog and Lbp features int the training set


[Xtr,Ytr,Xval,Yval,LabelsTrain,LabelsVal]= dataSplitPerImagePerPerson(X,y,perTrain,Labels);

clear X, clear y; %free memory
clear Xall;clear Yall;



%Normalization of the Hog features of the validation set
% Xval_norm = newFeatureNormalize(Xval,mu,sigma);

%% Initialize the data
NrC = size(unique(Ytr),2);%number of classes/names
k=1;
%compute the means of the classes
%M= computeMeans(Xtr_norm_projected,Ytr_norm,NrC);
M= computeMeans(Xtr,Ytr,NrC);

%% Runs a couple of SGD iterations
NrP=256; %desired number of the projections
[W,obj,W0] = NCM_train_sgd(Xtr,Ytr',M,NrP,Xval,Yval); %bug of the ncn

%% Evaluate performance on test dataset of W0 and W
%XtestNorm = newFeatureNormalize(Xtest,mu,sigma);
NrTop   = 5;
W0x     = W0 * Xtest;
W0m     = W0 * M;
d0      = sqdist(W0m,W0x);
p0      = softmax(-d0,1);
[~,lic] = sort(p0,1,'descend');
lic     = lic(1:NrTop,:)';
err0    = ilsvrc_eval_flat(lic,Ytest',NrTop);
fprintf('W0 |Test Error ');fprintf(' %7.3f ',err0*100);fprintf('\n');

Wx      = W * Xtest;
Wm      = W * M;
d       = sqdist(Wm,Wx);
p       = softmax(-d,1);
[~,lic] = sort(p,1,'descend');
lic     = lic(1:NrTop,:)';
[err,detail]     = ilsvrc_eval_flat(lic,Ytest',NrTop);
fprintf('W  |Test Error ');fprintf(' %7.3f ',err*100); fprintf('\n');

tEnd = toc(tStart);
fprintf('%d minutes and %f seconds\n',floor(tEnd/60),rem(tEnd,60));
%write results to a file
% S = struct('W', W);
% b = 'LearnedMatrix_Error_';
% d='Projections_';
% ext='.yml';
% c = sprintf('%s%d%s%d%s',b,err(1),d,NrP,ext); 
% cv.FileStorage(c,S);

%find the misclassified labels
missClassified=lic(:,1)==Ytest';
indexMiss=find(missClassified==0);
% %open file with write permission
 fid = fopen('missClassification256SameRandomKnnHogLbpMNPca2048.txt', 'w');
% %write a line of text
for j=1:size(indexMiss,1)
i=lic(indexMiss(j),1);
indicesImageClass = (find(Ytr==i));
Xcandids= Xtr(:,indicesImageClass);
LabelsCandids={};
for p=1:size(indicesImageClass,2)
    LabelsCandids{1,p}=LabelsTrain{1,indicesImageClass(p)};
end
x=(W*Xcandids)';
yq=(W*Xtest)';
[n,d]=knnsearch(x,yq(indexMiss(j),:));
fprintf(fid, '%s %d %s\n', LabelsTest{1,indexMiss(j)},lic(indexMiss(j),1),LabelsCandids{1,n});
end
 fclose(fid);
 
indexMiss=find(missClassified==1);
% %open file with write permission
 fid = fopen('corrClassification256SameRandomKnnHogLbpMNPca2048.txt', 'w');
% %write a line of text
for j=1:size(indexMiss,1)
i=lic(indexMiss(j),1);
indicesImageClass = (find(Ytr==i));
Xcandids= Xtr(:,indicesImageClass);
LabelsCandids={};
for p=1:size(indicesImageClass,2)
    LabelsCandids{1,p}=LabelsTrain{1,indicesImageClass(p)};
end
x=(W*Xcandids)';
yq=(W*Xtest)';
[n,d]=knnsearch(x,yq(indexMiss(j),:));
fprintf(fid, '%s %d %s\n', LabelsTest{1,indexMiss(j)},lic(indexMiss(j),1),LabelsCandids{1,n});
end
 fclose(fid);