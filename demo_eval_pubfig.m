clear all; close all;
%strat counting the time
tStart = tic;
addpath('C:\mexopencv-master\mexopencv-master');
% load data
fprintf('\nLoading images.\n');
load('featuresHohLbpGaborTanNARtUnitNormSep.mat','XallUnit');
load('featuresAllLabels.mat','Yall');
load('featuresFilenames.mat','Xfilenames');
removeCl=15;%11973 class 68
% Randomize and split the data X to training and test set
fprintf('\nSpliting to train and test set.\n');

XallUnitR=XallUnit(:,1:11973);YallR=Yall(1:11973);XfilenamesR=Xfilenames(1:11973);
[X,y,Xtest,Ytest,LabelsTest,Labels]= dataSplitTrainTest(XallUnitR,YallR,10,XfilenamesR);
clear XallUnit;
% Randomize and split the data X to training and validation set
perTrain=0.8;
fprintf('\nSpliting to train and validation set.\n');
[Xtr,Ytr,Xval,Yval,LabelsTrain,LabelsVal]= dataSplitPerImagePerPerson(X,y,perTrain,Labels);
clear X;
%Pca dimensionality reduction
fprintf('\nPca started\n');
varDesired=0.95;
model = pca(Xtr,varDesired);
[XtrPca ] = cvPcaProj(Xtr, model.W, model.mu);
clear Xtr;
%% Training the NCM
NrC = size(unique(Ytr),2);%number of classes/names
%compute the means of the classes
M= computeMeans(XtrPca,Ytr,NrC);
%Runs a couple of SGD iterations
NrP=256; %desired number of the projections
[W,obj,W0] = NCM_train_sgd(XtrPca,Ytr',M,NrP,Xval,Yval,model); %bug of the ncn

%% Evaluate performance on test dataset of W0 and W
%Project to pca the test data
[XtestPca ] = cvPcaProj(Xtest, model.W, model.mu);
NrTop   = 5;
W0x     = W0 * XtestPca;
W0m     = W0 * M;
d0      = sqdist(W0m,W0x);
p0      = softmax(-d0,1);
[~,lic] = sort(p0,1,'descend');
lic     = lic(1:NrTop,:)';
err0    = ilsvrc_eval_flat(lic,Ytest',NrTop);
fprintf('W0 |Test Error ');fprintf(' %7.3f ',err0*100);fprintf('\n');

Wx      = W * XtestPca;
Wm      = W * M;
d       = sqdist(Wm,Wx);
p       = softmax(-d,1);
[~,lic] = sort(p,1,'descend');
lic     = lic(1:NrTop,:)';
[err,detail]     = ilsvrc_eval_flat(lic,Ytest',NrTop);
fprintf('W  |Test Error ');fprintf(' %7.3f ',err*100); fprintf('\n');

tEnd = toc(tStart);
totalTime=(floor(tEnd/60));
fprintf('%d minutes and %f seconds\n',totalTime);
%% Store results for further process
folderName='experiment7/';
mkdir(folderName);
%save the learned matrix to a file
S = struct('W', W);
b = 'LearnedMatrix';
ext='.yml';
c = sprintf('%s%s',b,ext); 
c=[folderName c];
cv.FileStorage(c,S);

%save the accuracy to file
fid = fopen([folderName 'report.txt'], 'w');
fprintf(fid,'%s %s %s %s\n','Projections','top1error','top5error','totaltime'); 
fprintf(fid,'%d %f %f %f',NrP,err(1),err(5),totalTime); 
fclose(fid);

%find the misclassified labels
missClassified=lic(:,1)==Ytest';
indexMiss=find(missClassified==0);
% %open file with write permission
 fid = fopen([folderName 'missClassification.txt'], 'w');
% %write a line of text
for j=1:size(indexMiss,1)
i=lic(indexMiss(j),1);
indicesImageClass = (find(Ytr==i));
Xcandids= XtrPca(:,indicesImageClass);
LabelsCandids={};
for p=1:size(indicesImageClass,2)
    LabelsCandids{1,p}=LabelsTrain{1,indicesImageClass(p)};
end
x=(W*Xcandids)';
yq=(W*XtestPca)';
[n,d]=knnsearch(x,yq(indexMiss(j),:));
fprintf(fid, '%s %d %s\n', LabelsTest{1,indexMiss(j)},lic(indexMiss(j),1),LabelsCandids{1,n});
end
 fclose(fid);
 
indexMiss=find(missClassified==1);
% %open file with write permission
 fid = fopen([folderName 'corrClassification.txt'], 'w');
% %write a line of text
for j=1:size(indexMiss,1)
i=lic(indexMiss(j),1);
indicesImageClass = (find(Ytr==i));
Xcandids= XtrPca(:,indicesImageClass);
LabelsCandids={};
for p=1:size(indicesImageClass,2)
    LabelsCandids{1,p}=LabelsTrain{1,indicesImageClass(p)};
end
x=(W*Xcandids)';
yq=(W*XtestPca)';
[n,d]=knnsearch(x,yq(indexMiss(j),:));
fprintf(fid, '%s %d %s\n', LabelsTest{1,indexMiss(j)},lic(indexMiss(j),1),LabelsCandids{1,n});
end
 fclose(fid);




