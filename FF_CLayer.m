function feature_extracted = FF_CLayer(images, stride, zeropad, w,  b)

    %Calculate the dimensions of the resulting feature matrix extracted
    dim_hor = floor((size(images,1) - size(w,1) + 2*zeropad)/stride) + 1;
    dim_ver = floor((size(images,2) - size(w,1) + 2*zeropad)/stride) + 1;
    %The dimension of the feature matrix that is extracted
    feature_extracted =zeros( dim_hor, dim_ver, size(w,4), size(images,4));
    
    %Do zeropadding
    if zeropad~=0
        img_zeropad =  padarray(images,[zeropad zeropad],0,'both');
    else
        img_zeropad = images;
    end
    
    %Forward propogation for feature extraction
    for l = 1:size(images,4)
        temp_image = img_zeropad(:,:,:,l);
        for i = 1:dim_hor                           
            for j = 1:dim_ver
                for k = 1:size(w,4)
                    %Position of the slice from the image
                    temp_slice = temp_image(i*stride:i*stride + size(w,1) -1, j*stride:j*stride + size(w,1) -1, :);
%                     %Convolve the filter with the patch. Use Matrix
%                     multiplication instead
%                     filter = rot90(squeeze(w(:,:,:,k)),2);
%                     feature_extracted( i, j, k, l) = convn(temp_slice, filter, 'valid') + b(:,:,:,k);
                    v = temp_slice.* w(:,:,:,k);
                    feature_extracted(i,j,k,l) = sum(v(:)) + b(:,:,:,k);
                end
            end
        end
    end
 
end
    