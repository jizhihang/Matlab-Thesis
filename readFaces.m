function [ X,y ] = readFaces( faces )
%UNTITLED Summary of this function goes here
%   Read faces images generated from opencv yml format.

width= size(faces.TestImages{1},2);
height=size(faces.TestImages{1},1);
X = zeros(width*height,size(faces.TestImages,2));
y= zeros(size(faces.LabelImages));
for i=1:size(faces.TestImages,2)
    t = reshape(faces.TestImages{i},width*height,1);
    X(:,i)=t;
    y(1,i)= faces.LabelImages{i}+1;
end
end

