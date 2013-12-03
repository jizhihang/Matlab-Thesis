function [ rec,pr,tpr,fpr ] = computePR( scoreTest,labelTest,scoreDistract,YallTestLabels )
count=1;
for i=1:size(scoreTest,1)
confTest(i)=max(scoreTest(i,:));
end
for i=1:size(scoreDistract,1)
confDistract(i)=max(scoreDistract(i,:));
end
rec=zeros;
pr=zeros;
tpr=zeros;
fpr=zeros;
for th =0:0.001:1

%compute the labels assigned to test set
indexT=find(confTest>=th);
%compute recall from all test set how many were assigned a label
rec(count)=length(indexT)/size(scoreTest,1);



%compute precision

%find values above threshold

newlicTest=(labelTest(indexT)); 
newYall=YallTestLabels(indexT);
predictions=(newlicTest'==newYall);
cnt=find(predictions==1);
correctPred=length(cnt);
%take into account the distractors
indexD=find(scoreDistract>=th);

pr(count)=correctPred/(length(predictions)+length(indexD)+eps);
tpr(count)=correctPred/size(scoreTest,1);
fpr(count)=length(indexD)/size(scoreDistract,1);
count=count+1;
end