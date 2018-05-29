function [output]= forward_pooling(input, filter_size,stride)
    
    [height, width, channels,sampleNo] = size(input);
    

    new_height = floor(1 + (height - filter_size) / stride);
    new_width = floor(1 + (width - filter_size) / stride);
    new_channels = channels;
    
   
    output = zeros(new_height, new_width, new_channels,sampleNo);             
    

    for i =1:sampleNo                        
        for h =1:new_height                   
            for w =1:new_width                
                for c=1:new_channels           
                    
                   
                    top_pxl = h * stride;
                    bottom_pxl = top_pxl + filter_size;
                    leftmost_pxl = w * stride;
                    rightmost_pxl = leftmost_pxl + filter_size;
                    
                   
                    temp_slice = input(top_pxl:bottom_pxl-1, leftmost_pxl:rightmost_pxl-1, c,i);
                    
                    

                    output(h, w, c, i) = max(temp_slice(:));

                    
                end
            end
        end
    end
    
end