%% Clear everything for sanity check

clear all; close all;
tStart = tic;

% load data
fprintf('\nLoading mouth images.\n');
%path to mexopen cv to load files with open cv
addpath('C:\mexopencv-master\mexopencv-master');
X=[];
%% load function files from subfolders aswell
addpath (genpath ('.'));
% right_region = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\faces\faceImages_faces_0_30_LN_right_eye_HOG.yml');
%% Read the faces Images
% [X1,y] = readFaces(right_region);
% clear right_region;%free memory
% left_region = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\faces\faceImages_faces_0_30_LN_left_eye_HOG.yml');
% X=X1;
% clear X1;
% %% Read the faces Images
% [X2,y] = readFaces(left_region);
%  clear left_region; %free memory
%  X=[X;X2];
%  clear X2;
% % [X y width height names] = read_images('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\orl_faces');
left_cheek = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\faces\faceImages_faces_0_30_LN_left_cheek_HOG.yml');
%% Read the faces Images
[X3,y] = readFaces(left_cheek);
clear left_cheek;%free memory
X=[X;X3];
% 
clear X3;
% % [X y width height names] = read_images('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\orl_faces');
right_cheek = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\faces\faceImages_faces_0_30_LN_right_cheek_HOG.yml');
%% Read the faces Images
[X4,y] = readFaces(right_cheek);
clear right_cheek;%free memory
X=[X;X4];

% 
clear X4;
% [X y width height names] = read_images('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\orl_faces');
nose = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\faces\faceImages_faces_0_30_LN_nose_HOG_756.yml');
%% Read the faces Images
[X5,y] = readFaces(nose);
clear nose;%free memory
X=[X;X5];
% 
clear X5;
% [X y width height names] = read_images('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\orl_faces');
mouth = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\faces\faceImages_faces_0_30_LN_mouths_HOG_756.yml');
%% Read the faces Images
[X6,y] = readFaces(mouth);
clear mouth;%free memory
X=[X;X6];
clear X6;

% matnorm=repmat(sum(X,1),size(X,1),1);%code to scale the data to sum up to 1
% scaledX=X./matnorm;

% Randomize and split the data X to training and validation set
perTrain=0.6;
[Xtr,Ytr,Xval,Yval]= dataSplitPerImagePerPerson(X,y,perTrain);
%Normalize the labels from 1,4,45 to 1,2,3...
Ytr_norm=normalizeLabels(Ytr);
Yval_norm=normalizeLabels(Yval);
clear X, clear y; %free memory


%Standarization of the Hog features
% [Xtr_norm,mu,sigma]= featureNormalize(Xtr);

%Normalization of the Hog features of the validation set
% Xval_norm = newFeatureNormalize(Xval,mu,sigma);

%% Initialize the data
NrC = size(unique(Ytr_norm),2);%number of classes/names
k=1;
%compute the means of the classes
%M= computeMeans(Xtr_norm_projected,Ytr_norm,NrC);
M= computeMeans(Xtr,Ytr_norm,NrC);
count=0;
for p=1:1000
    %% Runs a couple of SGD iterations
    count=count+1;
    NrP=p; %desired number of the projections
    [W,obj,W0] = NCM_train_sgd(Xtr,Ytr_norm',M,NrP,k); %bug of the ncn
    
    
    %% Evaluate performance (on trainset) of W0 and W
    NrTop   = 5;
    
    %Evaluation
    W0x     = W0 * Xtr;
    W0m     = W0 * M;
    d0      = sqdist(W0m,W0x);
    p0      = softmax(-d0,1);
    [~,lic] = sort(p0,1,'descend');
    lic     = lic(1:NrTop,:)';
    err0    = ilsvrc_eval_flat(lic,Ytr_norm',NrTop);
    fprintf('W0 |Training Error ');fprintf(' %7.3f ',err0*100);fprintf('\n');
    
    Wx      = W * Xtr;
    Wm      = W * M;
    d       = sqdist(Wm,Wx);
    p       = softmax(-d,1);
    [~,lic] = sort(p,1,'descend');
    lic     = lic(1:NrTop,:)';
    err     = ilsvrc_eval_flat(lic,Ytr_norm',NrTop);
    fprintf('W  |Training Error ');fprintf(' %7.3f ',err*100); fprintf('\n');
    error_train(count,:)= err;
    
    W0x     = W0 * Xval;
    W0m     = W0 * M;
    d0      = sqdist(W0m,W0x);
    p0      = softmax(-d0,1);
    [~,lic] = sort(p0,1,'descend');
    lic     = lic(1:NrTop,:)';
    err0    = ilsvrc_eval_flat(lic,Yval_norm',NrTop);
    fprintf('W0 |Validation Error ');fprintf(' %7.3f ',err0*100);fprintf('\n');
    
    Wx      = W * Xval;
    Wm      = W * M;
    d       = sqdist(Wm,Wx);
    p       = softmax(-d,1);
    [~,lic] = sort(p,1,'descend');
    lic     = lic(1:NrTop,:)';
    err     = ilsvrc_eval_flat(lic,Yval_norm',NrTop);
    fprintf('W  |Validation Error ');fprintf(' %7.3f ',err*100); fprintf('\n');
    error_val(count,:)= err;
    param(count,1) = NrP;
end

minimum_error5=min(error_val(:,5));
index_best5=find(error_val(:,5)==minimum_error5);
C=index_best5;
if size(index_best5,1)>1
    minimum_error1=min(error_val(index_best5,1));
index_best1=find(error_val(index_best5,1)==minimum_error1);
 C= index_best5(index_best1);
end
best_param=(param(C,:));
 tEnd = toc(tStart);
 fprintf('%d minutes and %f seconds\n',floor(tEnd/60),rem(tEnd,60));















