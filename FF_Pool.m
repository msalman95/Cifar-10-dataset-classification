function pooled_features = FF_Pool(features, filterSize,stride)

    %Calculate the new dimensions of the pooled output
    dim_hor = floor(1 + (size(features,1) - filterSize) / stride);
    dim_ver = floor(1 + (size(features,2) - filterSize) / stride);
    %Create a zeros matrix of the output dimension
    pooled_features = zeros(dim_hor, dim_ver, size(features,3),size(features,4));             
    
    %Calculate the forward pooling matrix
    for l=1:size(features,4)
        for i=1:dim_hor                           
            for j=1:dim_ver
                for k=1:size(features,3)
                    %Get the slice from the convolution feature and choose
                    %the max one for max pooling
                    slice = features(i*stride-1:i*stride + filterSize -2, j*stride-1:j*stride + filterSize -2, k, l);
                    pooled_features(i, j, k, l) = max(slice(:));
                end
            end
        end
    end
    
end