function [bestMetric, obj, W0] = NCM_train_sgd_v2(Xtr,Ytr,Means,W,Xval,Yval,model)
% Basic SGD training for NCM objective
%
% Input
%   Xtr       [D x N]             Feature matrix D dimensional feature, N images
%   Ytr       [1 x N]             Ground Truth Class Labels for N images
%   Means   [D x C]             Means matrix, D dimensional, C classes
%   W       [d x D]             W matrix init
%         or scalar             Number of dimensions d
%  Xval       [D x n]             Feature matrix D dimensional feature, n images
%
% Output
%   W       [d x D]             Learned W matrix
%   obj     [I x 2]             Objective over I iterations
%   W0      [d x D]             Initial W matrix
%
% This functions implements a very basic SGD scheme to maximize the NCM objective, eq 7 in [1].
% It relies on the NCM_gradient_LogLoss_Eucl_k1 function.
%
% This function should be seen as a skeleton for a more elaborative function for evaluation.
% It lacks the cross validation of the number of dimensions or the number of iterations on a seperate validation set.
% And, also the possibility to store current results etc.
%
% See also: sqdist, softmax, NCM_gradient_LogLoss_Eucl_k1
%
% References
% [1]   Metric Learning for Large Scale Image Classification: Generalizing to New Classes at Near-Zero Cost,
%       Thomas Mensink, Jakob Verbeek, Florent Perronnin, Gabriela Csurka
%       In European Conference on Computer Vision (ECCV), 2012.
%       http://hal.inria.fr/hal-00722313/en

% Thomas Mensink, 2012-2013
% LEAR - INRIA Rhone Alpes, Grenoble, France
% ISLA - University of Amsterdam, Amsterdam, the Netherlands
% thomas.mensink@uva.nl

v = .51;
fprintf('%15s | v %7.4f | start %s\n',mfilename,v,datestr(now,31));

%Some useful sizes
[NrD,NrN]   = size(Xtr);
NrC         = size(Means,2);


%% Important variables which should be adjusted to the problem at hand
ops.iter1   = 1;
ops.iter2   = 1000000;           %The number of iterations
NrNi        = NrC;          %Number of images used per class
Eta         = .033;          %Learning rate
rs          = RandStream('mt19937ar','seed',0);  % Initialize random seed for data sampling etc.

%% Initialisation and input check

if isscalar(W),
    NrP     = W;
    W       = randn(rs,NrP,NrD,'single')*.1;
else
    NrP     = size(W,1);
end

W0          = W;
assert(NrD == size(W,2))
assert(NrD == size(Means,1));
assert(NrN == size(Ytr,1));
assert(NrP == size(W,1));
assert(max(Ytr) <= NrC);

step=10000;
MetricCell=cell(ops.iter2/step,1);
error_train=ones(ops.iter2/step,5);
error_val=ones(ops.iter2/step,5);
count=0;
NrTop=5;
bestMetric=W0;
fprintf('\tNrN %6d | Proj dim %3d Feat dim %5d | NrC %4d | Img per Iter %4d\n',NrN,NrP,NrD,NrC,NrNi);

%% Start SGD
obj(ops.iter2,2)    = 0;
t0 = tic;
fprintf('\tRunning from iters %d --> %d\n',ops.iter1, ops.iter2);

for iter = ops.iter1 : ops.iter2;
    t1 = tic;
    
    %% Training Iteration i
    ii  = randi(rs,NrN,1,NrNi);                              % Random select n
    Yii = Ytr(ii);
    Xii = Xtr(:,ii);
    
    %% Gradient Computation
    [f, G] = NCM_Gradient_LogLoss_Eucl_k1(W,Xii,Yii,Means);  % Compute the Gradient
    %         [f, G] = NCM_Gradient_LogLoss_Eucl_kAll(W,Xii,Yii,Means,k);
    
    
    
    %% Error checking
    if ~isfinite(f),
        fprintf('error objective is not finite\n');
        return;
    end
    
    %% Update the projection matrix and save
    W = W - Eta * reshape(G,size(W));
    
    obj(iter,:) = [f toc(t1)];
    
    
    %% Validate the results every 10k iterations
    if mod(iter ,step)==0
        fprintf('\r\t%10d i| f %15.10f |',iter,f);
        fprintf('| mean obj %15.10f | cum time %8.4f |',mean(obj(1:iter,1),1),toc(t0));
        count=count+1;
        totalSteps2go=(ops.iter2/step)-count;
        tEnd=toc(t0);
        timePerIter=tEnd/count;
        estimTime=totalSteps2go*timePerIter;
        fprintf('\n Estimated time... %d minutes and %f seconds',floor(estimTime/60),rem(estimTime,60));
        MetricCell{count,1}=W;
        %Check the training error between random w and learnedone
        W0x     = W0 * Xtr;
        W0m     = W0 * Means;
        d0      = sqdist(W0m,W0x);
        p0      = softmax(-d0,1);
        [~,lic] = sort(p0,1,'descend');
        lic     = lic(1:NrTop,:)';
        err0    = ilsvrc_eval_flat(lic,Ytr,NrTop);
        fprintf('\n');
        fprintf('W0 |Training Error ');fprintf(' %7.3f ',err0*100);fprintf('\n');
        
        
        Wx      = W * Xtr;
        Wm      = W * Means;
        d       = sqdist(Wm,Wx);
        p       = softmax(-d,1);
        [~,lic] = sort(p,1,'descend');
        lic     = lic(1:NrTop,:)';
        err     = ilsvrc_eval_flat(lic,Ytr,NrTop);
        fprintf('W  |Training Error ');fprintf(' %7.3f ',err*100); fprintf('\n');
        error_train(count,:)= err;
        
        %Check the validation error between random W0 and learned W
        
        W0x     = W0 * Xval;
        W0m     = W0 * Means;
        d0      = sqdist(W0m,W0x);
        p0      = softmax(-d0,1);
        [~,lic] = sort(p0,1,'descend');
        lic     = lic(1:NrTop,:)';
        err0    = ilsvrc_eval_flat(lic,Yval',NrTop);
        fprintf('W0 |Validation Error ');fprintf(' %7.3f ',err0*100);fprintf('\n');
        
        Wx      = W * Xval;
        Wm      = W * Means;
        d       = sqdist(Wm,Wx);
        p       = softmax(-d,1);
        [~,lic] = sort(p,1,'descend');
        lic     = lic(1:NrTop,:)';
        err     = ilsvrc_eval_flat(lic,Yval',NrTop);
        fprintf('W  |Validation Error ');fprintf(' %7.3f ',err*100); fprintf('\n');
        error_val(count,:)=err;%top 1 error
        
        %compute the validation error
        top1error=min(error_val(:,1));
        index_best1=find(error_val(:,1)==top1error);
        C=index_best1;
        %in case of a tie choose the best top 5 error
        if size(index_best1,1)>1
            top5error=min(error_val(index_best1,5));
            index_best5=find(error_val(index_best1,5)==top5error);
            C= index_best1(index_best5(1));
        end
        bestMetric=MetricCell{C,1};
    end
end
fprintf('\nFinally done !\n');
end

