function imo = getBatch_unit(images, varargin)


opts.imageDir=[];
opts.averageImage=[];
opts.imageStd=[];
opts.set=[];
[opts, varargin] = vl_argparse(opts, varargin);


imgDir = opts.imageDir;
im_rgb=fullfile(imgDir,images);
im = vl_imreadjpeg(im_rgb) ;
im= cellfun(@(x) resize_wrapper(x,512), im, 'un', 0);

% if opts.set(1)==1
    for im_i=1:numel(im)
     im{im_i}=crop_im(im{im_i},448,448);
    end
% else
%     for im_i=1:numel(im)
%      im{im_i}=crop_im_center(im{im_i},448,448);
%     end
% end


imo = ( zeros(448, 448, 3,numel(images),  'single') ) ;
        
              
for i=1:numel(images)
    if size(im{i},3)==3
        imo(:,:,:,i)=im{i};
    else
        imo(:,:,:,i)=repmat(im{i},[1,1,3]);
    end
end


if rand>0.5
    imo=fliplr(imo);
end

imo = imo/255;
imo = bsxfun(@minus, imo, reshape(opts.averageImage, 1, 1, 3));
imo = bsxfun(@rdivide, imo, reshape(opts.imageStd, 1, 1, 3));

end
