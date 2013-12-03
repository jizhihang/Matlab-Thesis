%% Clear everything for sanity check

clear all; close all;
tStart = tic;

% load data
fprintf('\nLoading images.\n');
%path to mexopen cv to load files with open cv
addpath('C:\mexopencv-master\mexopencv-master');
Xall=[];Xr=[];Xl=[];
%% load function files from subfolders aswell
addpath (genpath ('.'));
%% Read the faces Images
right_region = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures900_rightEye_60r_LN_train.yml');
[Xtrain,ytrain] = readFaces(right_region);
clear right_region;%free memory
right_region = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures900_rightEye_60r_LN_test.yml');
[Xtest,ytest] = readFaces(right_region);
clear right_region;%free memory
Xr=[Xtrain,Xtest];
Yall=[ytrain,ytest];
Xall=[Xall;Xr];
clear Xr;
clear Xtrain;clear Xtest;

%%Read the labels
filenames = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\faceImages_pubfiglfw_full_filenames_train.yml');
[Xfilenames1] = readLabels(filenames);
filenames = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\faceImages_pubfiglfw_full_filenames_test.yml');
[Xfilenames2] = readLabels(filenames);
Xfilenames=[Xfilenames1,Xfilenames2];
clear Xfilenames1;clear filenames2;
clear filenames;
%%Read the pithc and yaw values
pitchYaw1 = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\faceImages_pubfiglfw_LN_pitchYaw_train.yml');
[XpitchYaw1] = readPitchYaw(pitchYaw1);
pitchYaw2 = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\faceImages_pubfiglfw_LN_pitchYaw_test.yml');
[XpitchYaw2] = readPitchYaw(pitchYaw2);
XpitchYaw=[XpitchYaw1,XpitchYaw2];
clear XpitchYaw1;clear XpitchYaw2;
% get odd row indices
oddI = 1:2:size(XpitchYaw,2) ;
% get even row indices
evenI = 2:2:size(XpitchYaw,2) ;
Xpitch=XpitchYaw(oddI);
Xyaw=XpitchYaw(evenI);
Xall=[Xall;Xpitch];
Xall=[Xall;Xyaw];
clear Xpitch;clear Xyaw;
%% Read the faces Images
% left_region = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\faceImages_pubfiglfw_LN_lefteyeregion_HOG_324_train.yml');
left_region = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures900_leftEye_60r_LN_train.yml');
[Xtrain,y] = readFaces(left_region);
 clear left_region; %free memory
 left_region = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures900_leftEye_60r_LN_test.yml');
[Xtest,y] = readFaces(left_region);
 clear left_region; %free memory
 Xl=[Xtrain,Xtest];
 Xall=[Xall;Xl];
clear Xl;
clear Xtrain;clear Xtest;
%% Read the faces Images
% face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\faceImages_pubfiglfw_LN_ellipse_face_HOG_432_train.yml');
face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures324_face_60r_LN_train.yml');
[Xtrain,y] = readFaces(face);
clear face;%free memory
face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures324_face_60r_LN_test.yml');
[Xtest,y] = readFaces(face);
clear face;%free memory
Xf=[Xtrain,Xtest];
Xall=[Xall;Xf];
clear Xf;
clear Xtrain;clear Xtest;
%% Read the faces Images
nose = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures900_nose_60r_LN_train.yml');
[Xtrain,y] = readFaces(nose);
clear nose;%free memory
nose = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures900_nose_60r_LN_test.yml');
[Xtest,y] = readFaces(nose);
clear nose;%free memory
Xn=[Xtrain,Xtest];
Xall=[Xall;Xn];
clear Xn;
clear Xtrain;clear Xtest;
% [X y width height names] = read_images('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\orl_faces');
%% Read the faces Images
mouth = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures756_mouth_60r_LN_train.yml');
[Xtrain,y] = readFaces(mouth);
clear mouth;%free memory
mouth = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\publiclfw_hogFeatures756_mouth_60r_LN_test.yml');
[Xtest,y] = readFaces(mouth);
clear mouth;%free memory
Xm=[Xtrain,Xtest];
Xall=[Xall;Xm];
clear Xm;
clear Xtrain;clear Xtest;
% matnorm=repmat(sum(X,1),size(X,1),1);%code to scale the data to sum up to 1
% scaledX=X./matnorm;

%%Normalize the images to unit vectors
% Xnorm=bsxfun(@times,Xall,1./sqrt(sum(Xall.^2)));
%get 10 faces per person for test as in pubfig83 closed universe
[X,y,Xtest,Ytest,LabelsTest,Labels]= dataSplitTrainTest(Xall,Yall,10,Xfilenames);
% Randomize and split the data X to training and validation set
perTrain=0.85;
% y=normalizeLabels(y);
[Xtr,Ytr,Xval,Yval,LabelsTrain,LabelsVal]= dataSplitPerImagePerPerson(X,y,perTrain,Labels);
%Normalize the labels from 1,4,45 to 1,2,3...
% Ytr_norm=normalizeLabels(Ytr);
% Yval_norm=normalizeLabels(Yval);
clear X, clear y; %free memory
clear Xall;clear Yall;

%Standarization of the Hog features
% [Xtr_norm,mu,sigma]= featureNormalize(Xtr);

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
 fid = fopen('missClassification256SameRandomKnn.txt', 'w');
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
 fid = fopen('corrClassification256SameRandomKnn.txt', 'w');
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
 