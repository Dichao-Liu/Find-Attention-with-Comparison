function [ fn ] = getBatchWrapper(getBatch_opts)

fn = @(imdb,batch) getBatch(imdb,batch,getBatch_opts) ;
end

function [im,labels] = getBatch(imdb, batch, getBatch_opts)

opts.averageImage=getBatch_opts.averageImage;
opts.imageStd=getBatch_opts.imageStd;
opts.gpu=getBatch_opts.gpu;
opts.set = imdb.images.set(batch);


images = imdb.images.name(batch);
labels = imdb.images.label(batch) ;
im = getBatch_unit(images, opts, 'imageDir', imdb.imageDir) ;


if getBatch_opts.gpu
    im=gpuArray(im);
end



im = {'data', im(:,:,:,:,:),  'label' ,labels} ;


end

