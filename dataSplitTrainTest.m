function [ Xtr,Ytr,Xval,Yval,LabelsTest,LabelsTrain ] = dataSplitTrainTest( X , y, ntest,Xfilenames )
%DATASPLITPERIMAGEPERPERSON Summary of this function goes here
%   Detailed explanation goes here
Xval=[];
Xtr=[];
Yval=[];
Ytr=[];
numTrain = 0;
%Shuffle the data
no=clock;
% s = RandStream('mt19937ar','Seed',no(1,end));
s= RandStream('mt19937ar','seed',0);  % Initialize random seed for data sampling etc.
index = randsample(s,1:size(X,2), size(X,2));
X = X(:,index);
y = y(index);
for i=1:size(index,2)
    Xfilenames1{1,i}=Xfilenames{1,index(i)};
end
LabelsTest={};
LabelsTrain={};
%Split the data per person
for i=1:size(unique(y),2)
    indicesImageClass = (find(y==i));
    %check to see if indded we have ore than 4 images per person
    %numTrain = round(perTrain*size(indicesImageClass,2));
    numTrain = (size(indicesImageClass,2))-ntest;
    XvalClass = X(:,indicesImageClass(numTrain+1:end));
    YvalClass = y(indicesImageClass(numTrain+1:end));
     LabelsClassTest={};
    count=1;
    for j=numTrain+1:size(indicesImageClass,2)
        LabelsClassTest{1,count}=Xfilenames1{1,indicesImageClass(j)};
        count=count+1;
    end
    count=1;
    LabelsClassTrain={};
    for p=1:numTrain
        LabelsClassTrain{1,count}=Xfilenames1{1,indicesImageClass(p)};
        count=count+1;
    end
    %     LabelsClass=Xfilenames{1,indicesImageClass(numTrain+1:end)};
    XtrClass = X(:,indicesImageClass(1:numTrain));
    YtrClass = y(indicesImageClass(1:numTrain));
    Xval = [Xval XvalClass];
    Xtr = [Xtr XtrClass];
    Yval = [Yval YvalClass];
    LabelsTest = [LabelsTest, LabelsClassTest];
    LabelsTrain = [LabelsTrain, LabelsClassTrain];
    Ytr = [Ytr YtrClass];
end
end
