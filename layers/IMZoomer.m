classdef IMZoomer < dagnn.Layer
    
 properties
     out_size = 448;
 end
 
  methods
    function outputs = forward(obj, inputs, params)
      [outputs{1}] = imZoomer_fun(inputs{1},obj.out_size) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = imZoomer_fun(inputs{1},obj.out_size,derOutputs{1}) ;
      derParams = {} ;
    end
    function obj = IMZoomer(varargin)
      obj.load(varargin);
    end
  end
end
