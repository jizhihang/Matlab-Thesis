%%lib liner svm
clear all;close all;


%strat counting the time
tStart = tic;
addpath('C:\mexopencv-master\mexopencv-master');
addpath('C:\liblinear-1.94\matlab');
addpath('C:\libsvm-3.17\matlab');
%% load data
fprintf('\nLoading train images.\n');
%  load('featuresTrainHohLbpGaborTanNARtUnitNormSep.mat','XallTrainUnit');
X = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment5\TrainData.yml');

load('labelsTrain.mat','YallTrainLabels');
fprintf('\nLoading test images.\n');
load('featuresTestHohLbpGaborTanNARtUnitNormSep.mat','XallTestUnit');
load('labelsTest.mat','YallTestLabels');
fprintf('\nLoading distract images.\n');
load('featuresDistractHohLbpGaborTanNARtUnitNormSep.mat','XallDistractUnit');
load('labelsDistract.mat','YallDistractLabels');
fprintf('\nLoading pca projection.\n');
model = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment5\Pca512.yml');
% load('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment2\Pca.mat','model');
% fprintf('\nLoading learned matrix.\n');
% W = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment5\LearnedMatrix.yml');

load('C:\Users\Liveris\Documents\MasterThesis\faceClassifierNCMClean\experiment5\labelsTrain09Split.mat','Ytr');
%% Evaluate performance on test dataset of W0 and W

% NrC = size(unique(Ytr),2);%number of classes/names
% M= computeMeans(X.model,Ytr,NrC);
%  [MeansPca ] = cvPcaProj(M, model.model.W, model.model.mu);
%Project to pca the test data
[XtestPca ] = cvPcaProj(XallTestUnit, model.model.W, model.model.mu);
[XdistractPca ] = cvPcaProj(XallDistractUnit, model.model.W, model.model.mu);
[XtrainPca ] = cvPcaProj(X.model, model.model.W, model.model.mu);
XtrainPcaLib=XtrainPca';
XtrainPcaLibSp=sparse(XtrainPcaLib);

XtestPcaLib=XtestPca';
XtestPcaLibSp=sparse(XtestPcaLib);

XdistractPcaLib=XdistractPca';
XdistractPcaLibSp=sparse(XdistractPcaLib);

 modelSvm = svmtrain(Ytr', XtrainPcaLib, ' -c 4 -e 0.1 -b 1');
% [predict_label_test, accuracy_test, dec_values_test] = predict(YallTestLabels', XtestPcaLibSp, modelLib,'-b 1');
% model = ovrtrain(trainY, trainX, '-c 8 -g 4');
% modelSvm = ovrtrain( XtrainPcaLib,Ytr', '-c 8 -t 0 -e 0.1 -m 800');
[predict_label_test, accuracy_test, prob_estimates_test] = svmpredict(YallTestLabels', XtestPcaLib, modelSvm, '-b 1');
%fake distract labels
YallDistractLabels=ones(1,size(XdistractPcaLib,1));
[predict_label_distract, accuracy_distract, prob_estimates_distract] = svmpredict(YallDistractLabels', XdistractPcaLib, modelSvm, '-b 1');

[ recall,precision,tpr,fpr ] =computePRKnn(prob_estimates_test,predict_label_test,prob_estimates_distract,YallTestLabels );
recall256=recall;precision256=precision;tpr256=tpr;fpr256=fpr;
%compute the means of the train projected classes
% NrTop=5;
% 
% Wtest      = W.W * XtestPca;
% Wm      = W.W * MeansPca;
% dTest       = sqdist(Wm,Wtest);
% pTest       = softmax(-dTest,1);
% [~,licTest] = sort(pTest,1,'descend');
% licTest     = licTest(1:NrTop,:)';
% [err,detail]     = ilsvrc_eval_flat(licTest,YallTestLabels',NrTop);
% 
% %Project to pca the test data
% [XdistractPca ] = cvPcaProj(XallDistractUnit, model.model.W, model.model.mu);
% 
% Wdistract      = W.W * XdistractPca;
% dDistract       = sqdist(Wm,Wdistract);
% pDistract       = softmax(-dDistract,1);
% [~,licDistract] = sort(pDistract,1,'descend');
% licDistract     = licDistract(1:NrTop,:)';

figure(3)
% hlines=plot(recall256*100,precision256*100),recall512*100,precision512*100,recall1024*100,precision1024*100);
% set(hlines(1),'Displayname','NCM256')
% set(hlines(2),'Displayname','NCM512')
% set(hlines(3),'Displayname','NCM1024')
% legend(hlines);
plot(recall256*100,precision256*100)
title('Svm linear')
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
title('Svm linear')
xlabel('False Positive Rate (%)')
ylabel('True Positive Rate (%)')
grid on;
set(gca,'XTick',[0 10 20 30 40 50 60 70 80 90 100],...
'YTick',[0 10 20 30 40 50 60 70 80 90 100],...
'XLim',[0 100],...
'YLim',[0 100])


tEnd = toc(tStart);
totalTime=(floor(tEnd/60));
fprintf('%d minutes and %f seconds\n',totalTime);
