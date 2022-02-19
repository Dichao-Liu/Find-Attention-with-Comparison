function out_im=resize_wrapper(in_im,x)

if size(in_im,1)>size(in_im,2)
    out_im=imresize(in_im,[size(in_im,1)/(size(in_im,2)/x),x]);
else
    out_im=imresize(in_im,[x,size(in_im,2)/(size(in_im,1)/x)]);
end
    


