function[grad_output, del_weights, del_bias]=backward_convltn(grad_input, input, weights, padding)
%     """
%     Implement the backward propagation for a convolution function
%     
%     Arguments:
%     dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
%     cache -- cache of values needed for the conv_backward(), output of conv_forward()
%     
%     Returns:
%     dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
%                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
%     dW -- gradient of the cost with respect to the weights of the conv layer (W)
%           numpy array of shape (f, f, n_C_prev, n_C)
%     db -- gradient of the cost with respect to the biases of the conv layer (b)
%           numpy array of shape (1, 1, 1, n_C)
%     """


    [ height, width, channels,sampleNo] = size(input);
    
    
    [filter_size, ~,~, new_channels] = size(weights);

    [ new_height, new_width, ~,~] = size(grad_input);
    
    grad_output =zeros( height, width, channels,sampleNo);                           
    del_weights = zeros(filter_size, filter_size, channels, new_channels);
    del_bias = zeros(1, 1, 1, new_channels);

    if padding~=0
        input_padded =  padarray(input,[padding padding],0,'both');
        grad_input_padded = padarray(grad_output,[padding padding],0,'both');
    else
        input_padded = input;
        grad_input_padded = grad_output;
    end

    for i =1:sampleNo                       

        temp_input_padded = input_padded(:,:,:,i);
        temp_grad_padded = grad_input_padded(:,:,:,i);
        
        for h =1:new_height                   
            for w =1:new_width               
                for c =1:new_channels       
                
                    top_pxl = h;
                    bottom_pxl = top_pxl + filter_size;
                    leftmost_pxl = w;
                    rightmost_pxl = leftmost_pxl + filter_size;
                   
                    temp_slice = temp_input_padded(top_pxl:bottom_pxl-1, leftmost_pxl:rightmost_pxl-1, :);
                    
                    temp_grad_padded(top_pxl:bottom_pxl-1, leftmost_pxl:rightmost_pxl-1, :) =...
                        temp_grad_padded(top_pxl:bottom_pxl-1, leftmost_pxl:rightmost_pxl-1, :)+weights(: , :, :, c) * grad_input(h, w, c,i);
                    del_weights(:,:,:,c) = del_weights(:,:,:,c) + temp_slice * grad_input( h, w, c,i);
                    del_bias(:,:,:,c) = del_bias(:,:,:,c) + grad_input( h, w, c, i);
                end
            end
        end
    end
    if padding~=0
        grad_output( :, :, :,i) = temp_grad_padded(padding:(size(temp_grad_padded,1)-padding-1),...
            padding:(size(temp_grad_padded,2)-padding-1), :);
    else
        grad_output( :, :, :,i) = temp_grad_padded;
    end
       

   