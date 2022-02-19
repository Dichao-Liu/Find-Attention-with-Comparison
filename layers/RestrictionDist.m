classdef RestrictionDist < dagnn.Loss
  properties
    p = 2;
    aggregate = true;
    exp_l=0;
    exp_s=0.5;
    exp_a=0;
    thres_a=1*pi;
    thres=0.4;
  end

  methods
    function outputs = forward(obj, inputs, params)
        
      batch_size=size(inputs{1},4);
      translation=ones(1,1,2,batch_size,'single');
      if isa(inputs{1},'gpuArray')
          translation=gpuArray(translation);
      end
      
      ang_info=inputs{2}(:,:,4,:);
      rot_info=cat(3,cos(ang_info),-sin(ang_info),sin(ang_info),cos(ang_info));
      scale_info=inputs{1}(:,:,1:4,:);
      location_info=inputs{1}(:,:,5:6,:);
      
      exp_scale=rot_info*obj.exp_s;
      exp_location=obj.exp_l.*translation;
      exp_angle=obj.exp_a;
      
      scale_info(abs(scale_info-exp_scale)<obj.thres)=...
          exp_scale(abs(scale_info-exp_scale)<obj.thres);  
      location_info(abs(location_info-exp_location)<obj.thres)=...
          exp_location(abs(location_info-exp_location)<obj.thres);
      ang_info(abs(ang_info-exp_angle)<obj.thres_a)=exp_angle;
      
      loss_s = vl_nnpdist(scale_info, exp_scale, obj.p, [], 'aggregate', obj.aggregate, obj.opts{:}) ;
      loss_l = vl_nnpdist(location_info, exp_location, obj.p, [],'aggregate', obj.aggregate, obj.opts{:}) ;
      loss_a = vl_nnpdist(ang_info, exp_angle, obj.p, [],'aggregate', obj.aggregate, obj.opts{:}) ;
        
      
      outputs{1}=loss_s+loss_l+loss_a;
      obj.accumulateAverage(inputs, outputs);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1, numel(inputs));
      batch_size=size(inputs{1},4);
      translation=ones(1,1,2,batch_size,'single');
      if isa(inputs{1},'gpuArray')
          translation=gpuArray(translation);
      end
      
      ang_info=inputs{2}(:,:,4,:);
      rot_info=cat(3,cos(ang_info),-sin(ang_info),sin(ang_info),cos(ang_info));
      
      exp_scale=rot_info*obj.exp_s;
      exp_location=translation.*obj.exp_l;
      exp_angle=obj.exp_a;
      
      
      der_t = vl_nnpdist(inputs{1}(:,:,1:4,:), exp_scale, obj.p, derOutputs{1}, ...
            'aggregate', obj.aggregate, obj.opts{:}) ;
      der_l = vl_nnpdist(inputs{1}(:,:,5:6,:), exp_location, obj.p, derOutputs{1}, ...
            'aggregate', obj.aggregate, obj.opts{:}) ;
      der_a = vl_nnpdist(inputs{2}(:,:,4,:), exp_angle, obj.p, derOutputs{1}, ...
            'aggregate', obj.aggregate, obj.opts{:}) ;  
      
        
      der_t(abs(inputs{1}(:,:,1:4,:,:)-exp_scale)<obj.thres)=0;  
      der_l(abs(inputs{1}(:,:,5:6,:,:)-exp_location)<obj.thres)=0;
      der_a(abs(ang_info-exp_angle)<obj.thres_a)=exp_angle;
      
      
      derInputs{1}=cat(3,der_t,der_l);
      
      
      
      derInputs{2}=zeros(size(inputs{2}),'single');
      if isa(inputs{1},'gpuArray')
          derInputs{2}=gpuArray(derInputs{2});
      end
      derInputs{2}(:,:,4,:)=der_a;
      derParams = {} ;
    end

    function obj = RestrictionDist(varargin)
      obj.load(varargin) ;
      obj.loss = 'pdist';
    end
  end
end
