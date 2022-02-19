classdef PairwiseAffineGridGenerator < dagnn.Layer
%Adapted from DAGNN.AFFINEGRIDGENERATIOR ((c) 2016 Ankush Gupta)
%The gradient dA is clipped to be no larger than 1 and no less than -1

 properties
     Ho = 0;
     Wo = 0;
 end

  properties (Transient)
    xxyy ;
  end

  methods
    function outputs = forward(obj, inputs, ~)
      useGPU = isa(inputs{1}, 'gpuArray');

      A = inputs{1};
      nbatch = size(A,4);
      A = reshape(A, 2,3,nbatch);
      L = A(:,1:2,:);
      L = reshape(L,2,2*nbatch); 

      if isempty(obj.xxyy)
        obj.initGrid(useGPU);
      end

      t = A(:,3,:); 
      t = reshape(t,1,2*nbatch);
      g = bsxfun(@plus, obj.xxyy * L, t); 
      g = reshape(g, obj.Wo,obj.Ho,2,nbatch);
      
      xxyy_ = single(reshape(obj.xxyy, obj.Wo*obj.Ho,2));xxyy_ori=xxyy_;
      xxyy_=repmat(xxyy_,[1,1,nbatch]);
      xxyy_=round(xxyy_.*[obj.Wo,obj.Ho]/2);
      
      g_ = reshape(g, obj.Wo*obj.Ho,2,nbatch);
      g_=g_.*[obj.Wo,obj.Ho]/2;
      g_=cat(1,[floor(g_(:,1,:)),floor(g_(:,2,:))],...
          [floor(g_(:,1,:)),ceil(g_(:,2,:))],...
          [ceil(g_(:,1,:)),floor(g_(:,2,:))],...
          [ceil(g_(:,1,:)),ceil(g_(:,2,:))]);
      
      xxyy_ori=repmat(xxyy_ori,[1,1,nbatch]);
      
      for i=1:nbatch
          [~,inx,~]=intersect(xxyy_(:,:,i),g_(:,:,i),'rows');
          xxyy_ori(inx,:,i)=max(xxyy_ori(:))+1;
          
      end
      g_=xxyy_ori;
      g_ = reshape(g_, obj.Wo,obj.Ho,2,nbatch);

      g = permute(g, [3,2,1,4]);
      g_ = permute(g_, [3,2,1,4]);

      outputs = {g,g_};
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)

      useGPU = isa(derOutputs{1}, 'gpuArray');
      dY = derOutputs{1};
      nbatch = size(dY,4);
      dY = permute(dY, [3,2,1,4]);
      dA = zeros([2,3,nbatch], 'single');
      if useGPU, dA = gpuArray(dA); end
      dY = reshape(dY, obj.Ho*obj.Wo, 2*nbatch);
      dL = obj.xxyy' * dY;
      dL = reshape(dL,2,2,nbatch);
      dA(:,1:2,:) = dL;
      dt = reshape(sum(dY,1),2,1,nbatch);
      dA(:,3,:) = dt;
      dA = reshape(dA, size(inputs{1}));
      
      
      dY_ = derOutputs{2};
      dY_ = permute(dY_, [3,2,1,4]);
      dA_ = zeros([2,3,nbatch], 'single');
      if useGPU, dA_ = gpuArray(dA_); end
      dY_ = reshape(dY_, obj.Ho*obj.Wo, 2*nbatch);
      dL_ = obj.xxyy' * dY_;
      dL_ = reshape(dL_,2,2,nbatch);
      dA_(:,1:2,:) = dL_;
      dt_ = reshape(sum(dY_,1),2,1,nbatch);
      dA_(:,3,:) = dt_;
      dA_ = reshape(dA_, size(inputs{1}));
      
%       dA=(dA+dA_)/2;
      %%%%% clip gradient %%%%%%%
      dA(dA>1)=1;
      dA( dA<-1)=-1;
      derInputs = {dA};
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      nBatch = inputSizes{1}(4);
      outputSizes = {[2, obj.Ho, obj.Wo, nBatch]};
    end

    function obj = PairwiseAffineGridGenerator(varargin)
      obj.load(varargin) ;
      obj.Ho = obj.Ho ;
      obj.Wo = obj.Wo ;
      obj.xxyy = [] ;
    end

    function obj = reset(obj)
      reset@dagnn.Layer(obj) ;
      obj.xxyy = [] ;
    end

    function initGrid(obj, useGPU)
      xi = linspace(-1, 1, obj.Ho);
      yi = linspace(-1, 1, obj.Wo);
      [yy,xx] = meshgrid(xi,yi);
      xxyy = [yy(:), xx(:)] ;
      if useGPU
        xxyy = gpuArray(xxyy);
      end
      obj.xxyy = xxyy ;
    end

  end
end
