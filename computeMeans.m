function [ M ] = computeMeans( X,y,c )
%COMPUTEMEANS Summary of this function goes here
%   Detailed explanation goes here
% X DxN, data
% y 1xN, labels
% c, number of classes
M=[];
t=[X; y];
for i=1:c
    indices= find(t(end,:)==i);
    e=t(1:end-1,indices);
    a=mean(e,2);
    M=[M a];
end

end

