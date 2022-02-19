classdef PairContrastiveLoss < dagnn.Layer
% CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
%   Q query tuples, each packed in the form of (q,p,n1,..nN)
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

  properties
    margin = 0.7
    batch_list=[]
  end
  
  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      nbatch=size(inputs{1},4);
      if nbatch==1
          outputs{1}=0;
      else
          obj.batch_list=randperm(nbatch);
          if mod(nbatch,2)==0
              pairs=inputs{1}(:,:,:,obj.batch_list);
              label=repmat([-1,0],[1,nbatch/2]);
          else
              pairs=inputs{1}(:,:,:,obj.batch_list);
              label=repmat([-1,0],[1,(nbatch-1)/2]);
          end
      outputs{1} = cnn_contrastiveloss(pairs, label, obj.margin); % loss for this batch    
      nq = sum(gather(label) == -1); % number of query tuples in this batch
      n = obj.numAveraged; % number of query tuples done before this batch
      m = n + nq; % number of query tuples done so far
      obj.average = (n * obj.average + gather(outputs{1})) / m;
      obj.numAveraged = m;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
     useGPU = isa(inputs{1}, 'gpuArray');
     nbatch=size(inputs{1},4);
     derInputs{1}=zeros(size(inputs{1}),'single');
     if useGPU
        derInputs{1}=gpuArray(derInputs{1});
     end
     
     if nbatch~=1
         if mod(nbatch,2)==0
           pairs=inputs{1}(:,:,:,obj.batch_list);
           label=repmat([-1,0],[1,nbatch/2]);
         else
           pairs=inputs{1}(:,:,:,obj.batch_list);
           label=repmat([-1,0],[1,(nbatch-1)/2]);
         end
          [ders] = cnn_contrastiveloss(pairs, label, obj.margin, derOutputs{1});
          derInputs{1}(obj.batch_list)=ders;
     end
      derParams = {};
    end

    function reset(obj)
      obj.average = 0;
      obj.numAveraged = 0;
    end

    function obj = PairContrastiveLoss(varargin)
      obj.load(varargin);
    end
  end
end