
clear all;close all;
%strat counting the time
tStart = tic;
addpath('C:\mexopencv-master\mexopencv-master');
%% load data
fprintf('\nLoading train images.\n');
%  load('featuresTrainHohLbpGaborTanNARtUnitNormSep.mat','XallTrainUnit');
X = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment4\TrainData.yml');
% 
% load('labelsTrain.mat','YallTrainLabels');
% fprintf('\nLoading test images.\n');
load('featuresTestHohLbpGaborTanNARtUnitNormSep.mat','XallTestUnit');
load('labelsTest.mat','YallTestLabels');
fprintf('\nLoading distract images.\n');
load('featuresDistractHohLbpGaborTanNARtUnitNormSep.mat','XallDistractUnit');
load('labelsDistract.mat','YallDistractLabels');
fprintf('\nLoading pca projection.\n');
model = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment3\Pca512.yml');
% load('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment2\Pca.mat','model');
load('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment3\labelsTrainData.mat','Ytr');
%% Evaluate performance on test dataset of W0 and W

%Project to pca the test data
[XtestPca ] = cvPcaProj(XallTestUnit, model.model.W, model.model.mu);
%Project to pca the train data
[XtrainPca ] = cvPcaProj(X.model, model.model.W, model.model.mu);
%Project to pca the test data
[XdistractPca ] = cvPcaProj(XallDistractUnit, model.model.W, model.model.mu);

%L2 distance metric
NrTop=5;
M= computeMeans(XtrainPca,Ytr,83);
dTest       = sqdist(M,XtestPca);
pTest       = softmax(-dTest,1);
[~,licTest] = sort(pTest,1,'descend');
licTest     = licTest(1:NrTop,:)';

dDistract       = sqdist(M,XdistractPca);
pDistract       = softmax(-dDistract,1);
[~,licDistract] = sort(pDistract,1,'descend');
licDistract     = licDistract(1:NrTop,:)';
 [ recall,precision,tpr,fpr ] = computePR( pTest,licTest,pDistract,YallTestLabels );
recall256=recall;precision256=precision;tpr256=tpr;fpr256=fpr;
%compute the knn model
XtrainPcaKnn=XtrainPca';
XtestPcaKnn=XtestPca';
XdistractPcaKnn=XdistractPca';
YtrKnn=Ytr';
% mdl = ClassificationKNN.fit(XtrainPcaKnn,YtrKnn,'NumNeighbors',64);
% 
% [labelTest,scoreTest] = predict(mdl,XtestPcaKnn);
% [labelDistract,scoreDistract] = predict(mdl,XdistractPcaKnn);

%%second approach
dTest       = pdist2(XtestPcaKnn,XtrainPcaKnn);
[dTestSorted,indexTest] = sort(dTest,2,'ascend');
predictionTest = Ytr(indexTest(:,:));
ratio=zeros(size(dTestSorted,1),1);
%for each test sample
for i=1:size(predictionTest,1)
bestPred=predictionTest(i,1);
%find the first non nearest class neighbor
ix = find(predictionTest(i,:)~=bestPred, 1, 'first');
ratio(i)=dTestSorted(i,1)/dTestSorted(i,ix);
end
probEstimTest=1-ratio;


dDistract= pdist2(XdistractPcaKnn,XtrainPcaKnn);
[dDistractSorted,indexDistract] = sort(dDistract,2,'ascend');
predictionDistract = Ytr(indexDistract(:,:));
ratio=zeros(size(dDistractSorted,1),1);
%for each distract sample
for i=1:size(predictionDistract,1)
bestPred=predictionDistract(i,1);
%find the first non nearest class neighbor
ix = find(predictionDistract(i,:)~=bestPred, 1, 'first');
ratio(i)=dDistractSorted(i,1)/dDistractSorted(i,ix);
end
probEstimDistract=1-ratio;

% gdPred=Ytr(indexTest);
% minDistNeighbors=zeros(83,size(dTest,2));
% for j=1:size(dTestSorted,1)
%     for i=1:83
%    classLabels=find(prediction(j,:)==i);
%    %find the minimum dist neighbor of a specific class label
%    indexMinDist=find(min(dTestSorted(classLabels(1),j)));
%    %get the value
%    minDistNeighbors(i,j)=dTestSorted(classLabels(1,j));
%     end
% end
% %    pTest1       = softmax(-minDistNeighbors,1);
%   ratioTest=minDistNeighbors(:,1)./minDistNeighbors(:,2);
%   probTest=1-ratioTest;

[ recall,precision,tpr,fpr ] =computePRKnn(probEstimTest,predictionTest(:,1),probEstimDistract,YallTestLabels );
recall256=recall;precision256=precision;tpr256=tpr;fpr256=fpr;
figure(3)
% hlines=plot(recall256*100,precision256*100),recall512*100,precision512*100,recall1024*100,precision1024*100);
% set(hlines(1),'Displayname','NCM256')
% set(hlines(2),'Displayname','NCM512')
% set(hlines(3),'Displayname','NCM1024')
% legend(hlines);
plot(recall256*100,precision256*100)
title('1 Nearest Neighbor with best/secondBest dist confidence')
xlabel('Recall (%)')
ylabel('Precision (%)')
grid on;
set(gca,'XTick',[0 10 20 30 40 50 60 70 80 90 100],...
'YTick',[0 10 20 30 40 50 60 70 80 90 100],...
'XLim',[0 100],...
'YLim',[0 100])
% set(gca,XLim,[0 100])
% set(gca,YLim,[0 100])
figure(2)
% hlines=plot(fpr256*100,tpr256*100,fpr512*100,tpr512*100,fpr1024*100,tpr1024*100);
% set(hlines(1),'Displayname','NCM256')
% set(hlines(2),'Displayname','NCM512')
% set(hlines(3),'Displayname','NCM1024')
% legend(hlines);
plot(fpr256*100,tpr256*100)
title('1 Nearest Neighbor with best/secondBest dist confidence')
xlabel('False Positive Rate (%)')
ylabel('True Positive Rate (%)')
grid on;
set(gca,'XTick',[0 10 20 30 40 50 60 70 80 90 100],...
'YTick',[0 10 20 30 40 50 60 70 80 90 100],...
'XLim',[0 100],...
'YLim',[0 100])





