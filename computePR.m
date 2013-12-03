function [ rec,pr,tpr,fpr ] = computePR( pTest,licTest,pDistract,YallTestLabels )
count=1;
for i=1:size(pTest,2)
confTest(i)=max(pTest(:,i));
end
for i=1:size(pDistract,2)
confDistract(i)=max(pDistract(:,i));
end
rec=zeros;
pr=zeros;
tpr=zeros;
fpr=zeros;
for th =0:0.001:1

%compute the labels assigned to test set
indexT=find(confTest>=th);
%compute recall from all test set how many were assigned a label
rec(count)=length(indexT)/size(pTest,2);



%compute precision

%find values above threshold

newlicTest=(licTest(indexT,1)); 
newYall=YallTestLabels(indexT);
predictions=(newlicTest'==newYall);
cnt=find(predictions==1);
correctPred=length(cnt);
%take into account the distractors
indexD=find(confDistract>=th);

pr(count)=correctPred/(length(predictions)+length(indexD));
tpr(count)=correctPred/size(pTest,2);
fpr(count)=length(indexD)/size(pDistract,2);
count=count+1;
end

