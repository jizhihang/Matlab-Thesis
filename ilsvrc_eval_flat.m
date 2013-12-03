function [error, detail] = ilsvrc_eval_flat ( pred, gt, max_num_pred_per_image )
    % Evaluate flat error as used in the ILSVRC Challenge 2010
    %
    % Input
    %   Pred        I x Top     For each image the top class predictions
    %   gt          I x 1       The Ground truth class
    %   max_num     scalar      number of top images to consider
    %
    % Output
    %   flat_error  scalar
    %
    % This is a re-implementation of the provided function in the
    % ILSVRC 2011 v2 development kit [1]. The re-implementation allows the use
    % of input matrices for fast evaluation.
    %
    % References
    % [1]   The ImageNet large scale visual recognition challenge 2011,
    %       http://www.image-net.org/challenges/LSVRC/2011/
    %       http://www.image-net.org/challenges/LSVRC/2011/ILSVRC2011_devkit-2.0.tar.gz
    
    % Thomas Mensink, 2012-2013
    % LEAR - INRIA Rhone Alpes, Grenoble, France
    % ISLA - University of Amsterdam, Amsterdam, the Netherlands
    % thomas.mensink@uva.nl
    
    % Check input data
    if nargin < 3 || isempty(max_num_pred_per_image),       max_num_pred_per_image  = 5;         end    
    if size(pred,2) > max_num_pred_per_image                pred                    = pred(:,1:max_num_pred_per_image); end
       
    assert(size(pred,1)==size(gt,1),'size of prediction matrix and gt does not correspond');
    assert(size(gt,2) == 1,'assumed that gt contains only a single class per image');
    assert(all(gt > 0),'Error in GT some images do not have a label');
    
    % Compute the accuracy per rank 1...Top
    C   = cumsum(bsxfun(@eq,pred,gt),2);
    acc = mean(C,1);
    
    % Error is 1-accuracy
    error = 1-acc;
    detail= C;
end
    
