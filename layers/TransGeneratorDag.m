classdef TransGeneratorDag < dagnn.ElementWise
    
  properties
      s_factor=1
      l_factor=0.5
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = trans_generator(inputs{1},obj.s_factor, obj.l_factor) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = trans_generator(inputs{1},obj.s_factor, obj.l_factor,derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = TransGeneratorDag(varargin)
      obj.load(varargin) ;
    end
  end
end
