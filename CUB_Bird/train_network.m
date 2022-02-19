function train_network(varargin)


if ~isempty(gcp('nocreate'))
    delete(gcp)
end

run('..\matconvnet-1.0-beta25\matlab\vl_setupnn.m') ;
run('..\autonn-master\setup_autonn.m');
run('..\mcnExtraLayers-master\setup_mcnExtraLayers.m');
run('..\mcnPyTorch-master\setup_mcnPyTorch.m');
addpath('..\layers');

val_factor=5;
netname='bird-res50-biAtt-PRE';
opts.train.gpus = [1];

opts.train.learningRate =  1*[ 1e-3*ones(1,50/val_factor) ...
    1e-4*ones(1,45*2/val_factor) 1e-5*ones(1,1)] ;
opts.train.firstHalfOff = [zeros(1,50/val_factor),ones(1,45/val_factor),...
    zeros(1,45/val_factor),ones(1,45/val_factor),zeros(1,45/val_factor)] ;
opts.train.restrictionOff =  [zeros(size(opts.train.firstHalfOff)),1];




opts.train.expDir =[netname];
opts.train.numEpochs = 200;

[opts, varargin] = vl_argparse(opts, varargin) ;


opts.train.batchSize = 64;
opts.train.numSubBatches =16;
opts = vl_argparse(opts, varargin) ;


imdb=load('imdb_CUB.mat');
imdb.images.label=imdb.images.label(:,2);
imdb.images.set=imdb.images.set(:,2);
dataPath='F:\CUB_200_2011';



imdb.imageDir = fullfile(dataPath,imdb.imageDir);

opts.train.train = find(imdb.images.set==1) ;
opts.train.val = find(imdb.images.set==0); 
opts.train.train = repmat(opts.train.train,val_factor,1);
opts.train.train=opts.train.train(randperm(numel(opts.train.train)));


net=load([netname,'.mat']);

if isfield(net, 'net'), net=net.net;end

net =  dagnn.DagNN.loadobj(net);
net.rebuild() ;
opts.train.derOutputs = {} ;
for l=1:numel(net.layers)
    if (isa(net.layers(l).block, 'dagnn.Loss'))&& isempty(strfind(net.layers(l).block.loss, 'err'))...
            ||(isa(net.layers(l).block, 'PairwiseTripletLoss'))
  
      fprintf('setting derivative for layer %s \n', net.layers(l).name);
      opts.train.derOutputs = [opts.train.derOutputs, net.layers(l).outputs, {1}] ;
    end
end
net.meta.normalization.averageImage = gather(net.meta.normalization.averageImage);

net.conserveMemory = 1 ;

getBatch_opts.gpu=opts.train.gpus;
getBatch_opts.averageImage=net.meta.normalization.averageImage;
getBatch_opts.imageStd=net.meta.normalization.imageStd;

fn = getBatchWrapper(getBatch_opts) ;

[info] = cnn_train_dag_modified(net, imdb, fn, opts.train);


