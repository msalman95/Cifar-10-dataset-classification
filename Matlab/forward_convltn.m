function [output]= forward_convltn(input,weights, bias, stride,padding)

    [ height, width, channels, sampleNo] = size(input);
    [filter_size, ~,~, new_channels] = size(weights);    
    new_height = floor((height - filter_size + 2 * padding) / stride) + 1;
    new_width = floor((width - filter_size + 2 * padding) / stride) + 1;
    output =zeros( new_height, new_width, new_channels,sampleNo);
 
    if padding~=0
        input_padded =  padarray(input,[padding padding],0,'both');
    else
        input_padded = input;
    end
    
    for i=1:sampleNo
        temp_image = input_padded(:,:,:,i);
        for h=1:new_height                           
            for w=1:new_width
                for c=1:new_channels
                   
                    top_pxl = h * stride;
                    bottom_pxl = top_pxl + filter_size;
                    leftmost_pxl = w * stride;
                    rightmost_pxl = leftmost_pxl + filter_size;
                    
                    temp_slice = temp_image(top_pxl:bottom_pxl-1, leftmost_pxl:rightmost_pxl-1, :);
                    total=temp_slice.* weights(:,:,:,c);
                    output( h, w, c,i) = sum(total(:)) + bias(:,:,:,c);
                end
            end
        end
    end
end
    
