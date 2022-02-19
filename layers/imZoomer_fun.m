function [y] = imZoomer_fun(x,out_size, dzdy)
%Adapted from the code of DAGNN.AFFINEGRIDGENERATIOR,(c) 2016 Ankush Gupta.


xi = linspace(-1, 1, out_size);
yi = linspace(-1, 1, out_size);
[yy,xx] = meshgrid(xi,yi);
xxyy = [yy(:), xx(:)] ;
g = reshape(xxyy, [out_size,out_size,2]);
g = permute(g, [3,2,1,4]);
g = repmat(g,[1,1,1,size(x,4)]);
g=single(g);
if isa(x,'gpuArray')
    g=gpuArray(g);
end



if nargin <= 2 || isempty(dzdy)
    y=vl_nnbilinearsampler(x,g);
else
    [y,~]=vl_nnbilinearsampler(x,g,dzdy);
end