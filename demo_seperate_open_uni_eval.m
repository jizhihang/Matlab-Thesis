clear all;close all;
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


fprintf('\nLoading learned matrix.\n');
W = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment8sep\LearnedMatrix.yml');
modelHog = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment8sep\modelHog.yml');
modelLbp = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment8sep\modelLbp.yml');
modelGabor = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment8sep\modelGabor.yml');

XtrainHog=modelHog.modelHog.HogTrain;eHog=modelHog.modelHog.ePca;mHog=modelHog.modelHog.mPca;muHog=modelHog.modelHog.mu;
XtrainLbp=modelLbp.modelLbp.LbpTrain;eLbp=modelLbp.modelLbp.ePca;mLbp=modelLbp.modelLbp.mPca;muLbp=modelLbp.modelLbp.mu;
XtrainGabor=modelGabor.modelGabor.GaborTrain;eGabor=modelGabor.modelGabor.ePca;mGabor=modelGabor.modelGabor.mPca;muGabor=modelGabor.modelGabor.mu;


load('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment5\labelsTrain09Split.mat','Ytr');
XallTrain=[XtrainHog;XtrainLbp;XtrainGabor];
NrC = size(unique(Ytr),2);%number of classes/names
%compute the means of the classes
M= computeMeans(XallTrain,Ytr,NrC);

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

Wtest      = W.W * XallTest;
Wm      = W.W * M;
dTest       = sqdist(Wm,Wtest);
pTest       = softmax(-dTest,1);
[~,licTest] = sort(pTest,1,'descend');
licTest     = licTest(1:NrTop,:)';
[err,detail]     = ilsvrc_eval_flat(licTest,YallTestLabels',NrTop);


Wdistract      = W.W * XallDistract;
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
title('256NCM seperate Pca 512 dims per descriptor')
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
title('256NCM seperate Pca 512 dims per descriptor')
xlabel('False Positive Rate (%)')
ylabel('True Positive Rate (%)')
grid on;
set(gca,'XTick',[0 10 20 30 40 50 60 70 80 90 100],...
'YTick',[0 10 20 30 40 50 60 70 80 90 100],...
'XLim',[0 100],...
'YLim',[0 100])

