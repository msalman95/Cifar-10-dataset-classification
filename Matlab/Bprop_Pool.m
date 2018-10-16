function Jgrad_pool= Bprop_Pool(Jgrad_feature, feature, filterSize)
   %Zeros matrix for the gradient through 
   Jgrad_pool = zeros(size(feature));
    
    for l=1:size(feature,4)                   
        temp = feature(:,:,:,l);
        for i= 1:size(Jgrad_feature,1)
            for j=1:size(Jgrad_feature,2)
                for k=1:size(Jgrad_feature,3)
                    slice = temp(i:i + filterSize - 1, j:j + filterSize - 1, k);                    
                    mask = double(slice==max( slice (:)));
                        
                    Jgrad_pool(i:i + filterSize - 1, j:j + filterSize - 1, k, l) = ...
                    Jgrad_pool(i:i + filterSize - 1, j:j + filterSize - 1, k, l) + mask.* Jgrad_feature(i, j, k,l);
                end
            end
        end
    end
end

                   
                    
                       
                        
                   
