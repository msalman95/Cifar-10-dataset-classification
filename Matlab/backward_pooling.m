function Jgrad_pool= backward_pooling(Jgrad_feature, feature,filterSize)

    [ height, width, channels,sampleNo] = size(feature);
    [ new_height, new_width, new_channels ,sampleNo]= size(Jgrad_feature);
   
    Jgrad_pool = zeros(height, width, channels,sampleNo);
    
    for i=1:sampleNo                   
        temp_input = feature(:,:,:,i);
        for h= 1:new_height
            for w=1:new_width
                for c=1:new_channels
                    top_pxl = h;
                    bottom_pxl = top_pxl + filterSize;
                    leftmost_pxl = w;
                    rightmost_pxl = leftmost_pxl + filterSize;
                        temp_slice = temp_input(top_pxl:bottom_pxl-1, leftmost_pxl:rightmost_pxl-1, c);
                        
                        mask = double(temp_slice==max( temp_slice (:)));
                        
                        Jgrad_pool(top_pxl:bottom_pxl-1, leftmost_pxl:rightmost_pxl-1, c,i) = ...
                        Jgrad_pool(top_pxl:bottom_pxl-1, leftmost_pxl:rightmost_pxl-1, c,i) + mask .* Jgrad_feature(h, w, c,i);
                end
            end
        end
    end
end

                   
                    
                       
                        
                   
