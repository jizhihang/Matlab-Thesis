clear all; close all;
tStart = tic;

% load data
fprintf('\nLoading images.\n');
%path to mexopen cv to load files with open cv
addpath('C:\mexopencv-master\mexopencv-master');
XallUnit=[];
% % % 
 %% Read the faces Images
% % 
perTrain=0.9;
face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\hogFaceTanNArtTrain.yml');
[Xhog,y] = readFaces(face);
clear face;%free memory
fprintf('\nSpliting to train and validation set.\n');
[XhogTr,Ytr,XhogVal,Yval]= dataSplitPerImagePerPerson(Xhog,y,perTrain);

face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\lbpUniformFaceTanNArtTrain.yml');
[Xlbp,y] = readFaces(face);
clear face;%free memory
[XlbpTr,Ytr,XlbpVal,Yval]= dataSplitPerImagePerPerson(Xlbp,y,perTrain);

face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\gaborFaceTanNArtTrain.yml');
[Xgabor,y] = readFaces(face);
clear face;%free memory
[XgaborTr,Ytr,XgaborVal,Yval]= dataSplitPerImagePerPerson(Xgabor,y,perTrain);

%Test data
face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\hogFaceTanNArtTest.yml');
[XhogTest,YallTestLabels] = readFaces(face);


face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\lbpUniformFaceTanNArtTest.yml');
[XlbpTest,y] = readFaces(face);


face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\gaborFaceTanNArtTest.yml');
[XgaborTest,y] = readFaces(face);

%Distractors data
face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\hogFaceTanDistract.yml');
[XhogDistract,y] = readFaces(face);


face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\lbpUniformFaceTanDistract.yml');
[XlbpDistract,y] = readFaces(face);


face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\gaborFaceTanDistract.yml');
[XgaborDistract,y] = readFaces(face);


% load('labelsTrain.mat','YallTrainLabels');


%% Process the images
%Unit normalization
fprintf('\nHog.\n');
XhogUnitNorm=zeros(size(XhogTr));
for i=1:size(XhogTr,2)
    tempUnitNorm=createUnitVector(XhogTr(:,i));
    XhogUnitNorm(:,i)=tempUnitNorm;
end
%%Pca to 512 dimensions as paper
% M=512;
% [eHog, mHog, lambdaHog] = cvPca(XhogUnitNorm, M);
var=0.95;
modelPcaHog=pca(XhogUnitNorm,var);
eHog=modelPcaHog.W; mHog=modelPcaHog.mu;
[XtrHogPca ] = cvPcaProj(XhogUnitNorm, eHog, mHog);
%Zero meaned
[XtrHogZeroMeaned,muHog]=featureNormalize(XtrHogPca);
%Convert the val data also
XhogValUnitNorm=zeros(size(XhogVal));
for i=1:size(XhogVal,2)
    tempUnitNorm=createUnitVector(XhogVal(:,i));
    XhogValUnitNorm(:,i)=tempUnitNorm;
end
[XvalHogPca ] = cvPcaProj(XhogValUnitNorm, eHog, mHog);
XvalHogZeroMeaned=newFeatureNormalize(XvalHogPca,muHog);

%Unit normalization
fprintf('\nLbp.\n');
XlbpUnitNorm=zeros(size(XlbpTr));
for i=1:size(XlbpTr,2)
    tempUnitNorm=createUnitVector(XlbpTr(:,i));
    XlbpUnitNorm(:,i)=tempUnitNorm;
end
%%Pca to 512 dimensions as paper
% M=512;
% [eLbp, mLbp, lambdaLbp] = cvPca(XlbpUnitNorm, M);
modelPcaLbp=pca(XlbpUnitNorm,var);
eLbp=modelPcaLbp.W; mLbp=modelPcaLbp.mu;
[XtrLbpPca ] = cvPcaProj(XlbpUnitNorm, eLbp, mLbp);
%Zero meaned
[XtrLbpZeroMeaned,muLbp]=featureNormalize(XtrLbpPca);
%Convert the val data also
XlbpValUnitNorm=zeros(size(XlbpVal));
for i=1:size(XlbpVal,2)
    tempUnitNorm=createUnitVector(XlbpVal(:,i));
    XlbpValUnitNorm(:,i)=tempUnitNorm;
end
[XvalLbpPca ] = cvPcaProj(XlbpValUnitNorm, eLbp, mLbp);
XvalLbpZeroMeaned=newFeatureNormalize(XvalLbpPca,muLbp);

%Unit normalization
fprintf('\nGabor.\n');
XgaborUnitNorm=zeros(size(XgaborTr));
for i=1:size(XgaborTr,2)
    tempUnitNorm=createUnitVector(XgaborTr(:,i));
    XgaborUnitNorm(:,i)=tempUnitNorm;
end
%%Pca to 512 dimensions as paper
% M=512;
% [eGabor, mGabor, lambdaGabor] = cvPca(XgaborUnitNorm, M);
modelPcaGabor=pca(XgaborUnitNorm,var);
eGabor=modelPcaGabor.W; mGabor=modelPcaGabor.mu;
[XtrGaborPca ] = cvPcaProj(XgaborUnitNorm, eGabor, mGabor);
%Zero meaned
[XtrGaborZeroMeaned,muGabor]=featureNormalize(XtrGaborPca);
%Convert the val data also
XgaborValUnitNorm=zeros(size(XgaborVal));
for i=1:size(XgaborVal,2)
    tempUnitNorm=createUnitVector(XgaborVal(:,i));
    XgaborValUnitNorm(:,i)=tempUnitNorm;
end
[XvalGaborPca ] = cvPcaProj(XgaborValUnitNorm, eGabor, mGabor);
XvalGaborZeroMeaned=newFeatureNormalize(XvalGaborPca,muGabor);

XallTrain=[XtrHogZeroMeaned;XtrLbpZeroMeaned;XtrGaborZeroMeaned];
XallVal=[XvalHogZeroMeaned;XvalLbpZeroMeaned;XvalGaborZeroMeaned];

%% Train the images
% Randomize and split the data X to training and validation set

clear X;
%save the trainning file
folderName='experiment9sep/';
mkdir(folderName);
% S1 = struct('model', XallTrain);
% b1 = 'TrainDataZeroMeanedSeperate';
% ext1='.yml';
% c1 = sprintf('%s%s',b1,ext1); 
% c1=[folderName c1];
% cv.FileStorage(c1,S1);

%% Training the NCM
NrC = size(unique(Ytr),2);%number of classes/names
%compute the means of the classes
M= computeMeans(XallTrain,Ytr,NrC);
%Runs a couple of SGD iterations
NrP=256; %desired number of the projections
[W,obj,W0] = NCM_train_sgd_v2(XallTrain,Ytr',M,NrP,XallVal,Yval); %bug of the ncn

tEnd = toc(tStart);
totalTime=(floor(tEnd/60));
fprintf('%d minutes and %f seconds\n',totalTime);
%% Store results for further process

%save the learned matrix to a file
S = struct('W', W);
b = 'LearnedMatrix';
ext='.yml';
c = sprintf('%s%s',b,ext); 
c=[folderName c];
cv.FileStorage(c,S);

%save the pca matrix to a file
modelHog.HogTrain=XtrHogZeroMeaned;modelHog.ePca=eHog;modelHog.mPca=mHog;modelHog.mu=muHog;
S1 = struct('modelHog', modelHog);
b1='modelHog';
ext1='.yml';
c1 = sprintf('%s%s',b1,ext1); 
c1=[folderName c1];
cv.FileStorage(c1,S1);

modelLbp.LbpTrain=XtrLbpZeroMeaned;modelLbp.ePca=eLbp;modelLbp.mPca=mLbp;modelLbp.mu=muLbp;
S1 = struct('modelLbp', modelLbp);
b1='modelLbp';
ext1='.yml';
c1 = sprintf('%s%s',b1,ext1); 
c1=[folderName c1];
cv.FileStorage(c1,S1);

modelGabor.GaborTrain=XtrGaborZeroMeaned;modelGabor.ePca=eGabor;modelGabor.mPca=mGabor;modelGabor.mu=muGabor;
S1 = struct('modelGabor', modelGabor);
b1='modelGabor';
ext1='.yml';
c1 = sprintf('%s%s',b1,ext1); 
c1=[folderName c1];
cv.FileStorage(c1,S1);

%% Evaluate

XhogTestUnitNorm=zeros(size(XhogTest));
for i=1:size(XhogTest,2)
    tempUnitNorm=createUnitVector(XhogTest(:,i));
    XhogTestUnitNorm(:,i)=tempUnitNorm;
end
[XtestHogPca ] = cvPcaProj(XhogTestUnitNorm, eHog, mHog);
XtestHogZeroMeaned=newFeatureNormalize(XtestHogPca,muHog);

XlbpTestUnitNorm=zeros(size(XlbpTest));
for i=1:size(XlbpTest,2)
    tempUnitNorm=createUnitVector(XlbpTest(:,i));
    XlbpTestUnitNorm(:,i)=tempUnitNorm;
end
[XtestLbpPca ] = cvPcaProj(XlbpTestUnitNorm, eLbp, mLbp);
XtestLbpZeroMeaned=newFeatureNormalize(XtestLbpPca,muLbp);

XgaborTestUnitNorm=zeros(size(XgaborTest));
for i=1:size(XgaborTest,2)
    tempUnitNorm=createUnitVector(XgaborTest(:,i));
    XgaborTestUnitNorm(:,i)=tempUnitNorm;
end
[XtestGaborPca ] = cvPcaProj(XgaborTestUnitNorm, eGabor, mGabor);
XtestGaborZeroMeaned=newFeatureNormalize(XtestGaborPca,muGabor);


XhogDistractUnitNorm=zeros(size(XhogDistract));
for i=1:size(XhogDistract,2)
    tempUnitNorm=createUnitVector(XhogDistract(:,i));
    XhogDistractUnitNorm(:,i)=tempUnitNorm;
end
[XdistractHogPca ] = cvPcaProj(XhogDistractUnitNorm, eHog, mHog);
XdistractHogZeroMeaned=newFeatureNormalize(XdistractHogPca,muHog);

XlbpDistractUnitNorm=zeros(size(XlbpDistract));
for i=1:size(XlbpDistract,2)
    tempUnitNorm=createUnitVector(XlbpDistract(:,i));
    XlbpDistractUnitNorm(:,i)=tempUnitNorm;
end
[XdistractLbpPca ] = cvPcaProj(XlbpDistractUnitNorm, eLbp, mLbp);
XdistractLbpZeroMeaned=newFeatureNormalize(XdistractLbpPca,muLbp);

XgaborDistractUnitNorm=zeros(size(XgaborDistract));
for i=1:size(XgaborDistract,2)
    tempUnitNorm=createUnitVector(XgaborDistract(:,i));
    XgaborDistractUnitNorm(:,i)=tempUnitNorm;
end
[XdistractGaborPca ] = cvPcaProj(XgaborDistractUnitNorm, eGabor, mGabor);
XdistractGaborZeroMeaned=newFeatureNormalize(XdistractGaborPca,muGabor);

XallTest=[XtestHogZeroMeaned;XtestLbpZeroMeaned;XtestGaborZeroMeaned];
XallDistract=[XdistractHogZeroMeaned;XdistractLbpZeroMeaned;XdistractGaborZeroMeaned];
NrTop=5;

Wtest      = W * XallTest;
Wm      = W * M;
dTest       = sqdist(Wm,Wtest);
pTest       = softmax(-dTest,1);
[~,licTest] = sort(pTest,1,'descend');
licTest     = licTest(1:NrTop,:)';
[err,detail]     = ilsvrc_eval_flat(licTest,YallTestLabels',NrTop);


Wdistract      = W * XallDistract;
dDistract       = sqdist(Wm,Wdistract);
pDistract       = softmax(-dDistract,1);
[~,licDistract] = sort(pDistract,1,'descend');
licDistract     = licDistract(1:NrTop,:)';

 [ recall,precision,tpr,fpr ] = computePR( pTest,licTest,pDistract,YallTestLabels );
recall256=recall;precision256=precision;tpr256=tpr;fpr256=fpr;
figure(3)
% hlines=plot(recall256*100,precision256*100),recall512*100,precision512*100,recall1024*100,precision1024*100);
% set(hlines(1),'Displayname','NCM256')
% set(hlines(2),'Displayname','NCM512')
% set(hlines(3),'Displayname','NCM1024')
% legend(hlines);
plot(recall256*100,precision256*100)
title('256NCM seperate Pca 0.95 variance per descriptor')
xlabel('Recall (%)')
ylabel('Precision (%)')
grid on;
set(gca,'XTick',[0 10 20 30 40 50 60 70 80 90 100],...
'YTick',[0 10 20 30 40 50 60 70 80 90 100],...
'XLim',[0 100],...
'YLim',[0 100])

figure(2)
% hlines=plot(fpr256*100,tpr256*100,fpr512*100,tpr512*100,fpr1024*100,tpr1024*100);
% set(hlines(1),'Displayname','NCM256')
% set(hlines(2),'Displayname','NCM512')
% set(hlines(3),'Displayname','NCM1024')
% legend(hlines);
plot(fpr256*100,tpr256*100)
title('256NCM seperate Pca 0.95 variance per descriptor')
xlabel('False Positive Rate (%)')
ylabel('True Positive Rate (%)')
grid on;
set(gca,'XTick',[0 10 20 30 40 50 60 70 80 90 100],...
'YTick',[0 10 20 30 40 50 60 70 80 90 100],...
'XLim',[0 100],...
'YLim',[0 100])

