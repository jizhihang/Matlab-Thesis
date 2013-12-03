function [ y ] = normalizeLabels( y )
%Normalize the labels that have the format 3,5,7,...9 to 1,2,3,...6
if y(1,1)>1
    y(1,1:end)=y(1,1:end)-y(1,1)+1;
end
for i=1:length(y)-1
   if (y(1,i+1)-y(1,i)>1)
       y(1,(i+1):end)=y(1,(i+1):end)+1 -( y(1,i+1)-y(1,i));
   end
end   

end

