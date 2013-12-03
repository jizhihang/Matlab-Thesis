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
face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\hogFaceTanNArtTest.yml');
[Xhog,y] = readFaces(face);
clear face;%free memory

face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\lbpUniformFaceTanNArtTest.yml');
[Xlbp,y] = readFaces(face);
clear face;%free memory

face = cv.FileStorage('C:\Users\Liveris\Documents\MasterThesis\FaceDatasets\pubfig83lfw_raw_in_dirs\featuresPaper\gaborFaceTanNArtTest.yml');
[Xgabor,y] = readFaces(face);
clear face;%free memory




%% Process the images
%Unit normalization
XhogUnitNorm=zeros(size(Xhog));
for i=1:size(Xhog,2)
    tempUnitNorm=createUnitVector(Xhog(:,i));
    XhogUnitNorm(:,i)=tempUnitNorm;
end

%Unit normalization
XlbpUnitNorm=zeros(size(Xlbp));
for i=1:size(Xlbp,2)
    tempUnitNorm=createUnitVector(Xlbp(:,i));
    XlbpUnitNorm(:,i)=tempUnitNorm;
end

%Unit normalization
XgaborUnitNorm=zeros(size(Xgabor));
for i=1:size(Xgabor,2)
    tempUnitNorm=createUnitVector(Xgabor(:,i));
    XgaborUnitNorm(:,i)=tempUnitNorm;
end

XallUnit=[XallUnit;XlbpUnitNorm];
XallUnit=[XallUnit;XhogUnitNorm];
XallUnit=[XallUnit;XgaborUnitNorm];
%remove the gaps between the labels
% yNorm=normalizeLabels(y);
% %make them start from 84
% yFinal=yNorm+83;












