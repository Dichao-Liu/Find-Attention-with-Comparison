classdef PairwiseTripletLoss < dagnn.Layer
% TRIPLETLOSS layer that computes triplet loss for a batch of images:
%   Q query tuples, each packed in the form of (q,p,n1,..nN)
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 

  properties
    margin = 0.1
    batch_list=[]
    p1=[];
    p2=[];
    useGPU=0;
    ignoreAverage=0
  end
  
  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
       obj.useGPU = isa(inputs{1}, 'gpuArray');
       nbatch=size(inputs{1},4);
     if nbatch==1
          outputs{1}=0;
      else
          obj.batch_list=randperm(nbatch);
          if mod(nbatch,2)==0
              obj.p1=obj.batch_list(1:nbatch/2);
              obj.p2=obj.batch_list(nbatch/2+1:end);
          else
              obj.p1=obj.batch_list(1:floor(nbatch/2));
              obj.p2=obj.batch_list(ceil(nbatch/2):end-1);
          end
          
      [pairs,label] = obj.makePairs(inputs);
      outputs{1} = cnn_tripletloss(pairs, label, obj.margin);
      
      nq = sum(gather(label) == -1); % number of query tuples in this batch
      n = obj.numAveraged; % number of query tuples done before this batch
      m = n + nq; % number of query tuples done so far
      obj.average = (n * obj.average + gather(outputs{1})) / m;
      obj.numAveraged = m;
     end
      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
       nbatch=size(inputs{1},4);
       derInputs{1}=zeros(size(inputs{1}),'single');
       derInputs{2}=zeros(size(inputs{1}),'single');
       if obj.useGPU
        derInputs{1}=gpuArray(derInputs{1});
        derInputs{2}=gpuArray(derInputs{2});
       end
      
        
      if nbatch~=1  
          [pairs,label] = obj.makePairs(inputs);
          [der] = cnn_tripletloss(pairs, label, obj.margin, derOutputs{1});
          der_1=der(:,:,:,1:3*numel(obj.p1));der_2=der(:,:,:,3*numel(obj.p1)+1:end);
          der_in1_p1=der_1(:,:,:,1:3:end);
          der_in1_p2=der_1(:,:,:,2:3:end);
          der_in2_p1=der_1(:,:,:,3:3:end);
          
          der_in1_p2_=der_2(:,:,:,1:3:end);
          der_in1_p1_=der_2(:,:,:,2:3:end);
          der_in2_p2=der_2(:,:,:,3:3:end);
          
          der_in1=cat(4,(der_in1_p1+der_in1_p1_)/2,(der_in1_p2+der_in1_p2_)/2);
          der_in2=cat(4,der_in2_p1,der_in2_p2);
          
          derInputs{1}(:,:,:,[obj.p1,obj.p2])=der_in1;
          derInputs{2}(:,:,:,[obj.p1,obj.p2])=der_in2;
          
      end
      
      derParams = {};
    end
    
    function [pairs,label] = makePairs(obj,inputs)
        nbatch_half=numel(obj.p1);
        pairs=zeros(size(inputs{1},1),size(inputs{1},2),size(inputs{1},3),nbatch_half*3,'single');
      if obj.useGPU
          pairs=gpuArray(pairs); 
      end
      pairs2=pairs;
      
      pairs(:,:,:,1:3:end)=inputs{1}(:,:,:,obj.p1);
      pairs(:,:,:,2:3:end)=inputs{1}(:,:,:,obj.p2);
      pairs(:,:,:,3:3:end)=inputs{2}(:,:,:,obj.p1);
      
      pairs2(:,:,:,1:3:end)=inputs{1}(:,:,:,obj.p2);
      pairs2(:,:,:,2:3:end)=inputs{1}(:,:,:,obj.p1);
      pairs2(:,:,:,3:3:end)=inputs{2}(:,:,:,obj.p2);
      
      pairs=cat(4,pairs,pairs2);
      label=repmat([-1,1,0],[1,nbatch_half]);      
    end

    function reset(obj)
      obj.average = 0;
      obj.numAveraged = 0;
    end

    function obj = PairwiseTripletLoss(varargin)
      obj.load(varargin);
    end
  end
end