function[Jgrad_out, Jgrad_w, Jgrad_b]= Bprop_conv(Jgrad_feature, feature, weights, zeropad)
    %Size of the gradients
    Jgrad_out =zeros(size(feature));                           
    Jgrad_w = zeros(size(weights,1), size(weights,1),size(feature,3), size(weights,4));
    Jgrad_b = zeros(1, 1, 1, size(weights,4));

    if zeropad~=0
        input_zeropad =  padarray(feature,[zeropad zeropad],0,'both');
        Jgrad_zeropad = padarray(Jgrad_out,[zeropad zeropad],0,'both');
    else
        input_zeropad = feature;
        Jgrad_zeropad = Jgrad_out;
    end

    for l =1:size(feature,4)                       
        feature_zeropad = input_zeropad(:,:,:,l);
        grad_zeropad = Jgrad_zeropad(:,:,:,l);
        for i =1:size(Jgrad_feature,1)                   
            for j =1:size(Jgrad_feature,2)               
                for k =1:size(weights,4)       
                    %Get the slice from the feature matrix
                    slice = feature_zeropad(i:i + size(weights,1) - 1, j:j + size(weights,1) - 1, :);
                    %Calculate the gradients
                    grad_zeropad(i:i + size(weights,1) - 1, j:j + size(weights,1) - 1, :) = grad_zeropad(i:i + size(weights,1) - 1, j:j+ size(weights,1) - 1, :) + weights(: , :, :, k) * Jgrad_feature(i, j, k,l);
                    Jgrad_w(:,:,:,k) = Jgrad_w(:,:,:,k) + slice * Jgrad_feature( i, j, k,l);
                    Jgrad_b(:,:,:,k) = Jgrad_b(:,:,:,k) + Jgrad_feature( i, j, k, l);
                end
            end
        end
    end
    if zeropad~=0
        Jgrad_out( :, :, :,l) = grad_zeropad(zeropad:(size(grad_zeropad,1)-zeropad-1),zeropad:(size(grad_zeropad,2)-zeropad-1), :);
    else
        Jgrad_out( :, :, :,l) = grad_zeropad;
    end
end

   
