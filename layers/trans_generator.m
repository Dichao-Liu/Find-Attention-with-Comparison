function [y] = trans_generator(x, s_factor, l_factor, dzdy)

if nargin <= 3 || isempty(dzdy)
    scale_info=x(:,:,1,:);
    scale_info=tanh(scale_info)*s_factor;

    location_info=x(:,:,2:3,:);
    location_info=tanh(location_info)*l_factor;

    rot_info=x(:,:,4,:);
    rot_info=cat(3,cos(rot_info),-sin(rot_info),sin(rot_info),cos(rot_info));
    
    y=cat(3,scale_info.*rot_info,location_info);
else
    scale_info=tanh(x(:,:,1,:))*s_factor;
    rot_info=x(:,:,4,:);
    
    der_r=cat(3,-sin(rot_info),-cos(rot_info),cos(rot_info),-sin(rot_info));
    der_r=der_r.*scale_info.*dzdy(:,:,1:4,:);
    der_r=mean(der_r,3);
    
    rot_info=cat(3,cos(rot_info),-sin(rot_info),sin(rot_info),cos(rot_info));
    der_s=1-(tanh(x(:,:,1,:))).^2;
    der_s=s_factor*repmat(der_s,[1,1,4,1]).*rot_info.*dzdy(:,:,1:4,:);
    der_s=mean(der_s,3);
    
    der_l=1-(tanh(x(:,:,2:3,:))).^2;
    der_l=l_factor*der_l.*dzdy(:,:,5:6,:);
    
    y=cat(3,der_s,der_l,der_r);
end