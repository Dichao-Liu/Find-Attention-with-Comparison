classdef CenterDist < dagnn.Loss
  properties
    p = 2;
    aggregate = true;
  end

  methods
    function outputs = forward(obj, inputs, params)
        
      batch_size=size(inputs{1},4);
      center=mean(inputs{1},4);
      center=repmat(center,[1,1,1,batch_size]);
      outputs{1}=vl_nnpdist(inputs{1}, center, obj.p, [], 'aggregate', obj.aggregate, obj.opts{:}) ;
      obj.accumulateAverage(inputs, outputs);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      batch_size=size(inputs{1},4);
      center=mean(inputs{1},4);
      center=repmat(center,[1,1,1,batch_size]);
      
      
      derInputs{1} = vl_nnpdist(inputs{1}, center, obj.p, derOutputs{1}, ...
            'aggregate', obj.aggregate, obj.opts{:}) ;
       derParams = {} ;
    end

    function obj = CenterDist(varargin)
      obj.load(varargin) ;
      obj.loss = 'pdist';
    end
  end
end
