
clear all; close all;
%strat counting the time
tStart = tic;
addpath('C:\mexopencv-master\mexopencv-master');
% load data
fprintf('\nLoading training images.\n');
load('featuresTrainHohLbpGaborTanNARtUnitNormSep.mat','XallTrainUnit');
load('labelsTrain.mat','YallTrainLabels');

% Randomize and split the data X to training and validation set
perTrain=0.9;
fprintf('\nSpliting to train and validation set.\n');
[Xtr,Ytr,Xval,Yval]= dataSplitPerImagePerPerson(XallTrainUnit,YallTrainLabels,perTrain);
clear X;
%save the trainning file
folderName='experiment5/';
mkdir(folderName);
S1 = struct('model', Xtr);
b1 = 'TrainData';
ext1='.yml';
c1 = sprintf('%s%s',b1,ext1); 
c1=[folderName c1];
cv.FileStorage(c1,S1);
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
S1 = struct('model', model);
b1 = 'Pca512';
ext1='.yml';
c1 = sprintf('%s%s',b1,ext1); 
c1=[folderName c1];
cv.FileStorage(c1,S1);








