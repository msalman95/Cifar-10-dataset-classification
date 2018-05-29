function [w_c1, b_c1, w_c2, b_c2, w_c3, b_c3, w_fc1, b_fc1, w_fc2 b_fc2] = TrainCNN(features_train, features_test, labels_train, labels_test)

%Initialize the parameters here
epoch = 100;    %Maximum number of epochs
num_batch = 1000;   %Total number of batches
samples_batch = length(labels_train)/num_batch; %Samples in each batch
num_channels = 3;   %RGB - 3 channels one for each
filterSize = 3;   %size of filter - 3x3
filter_c1 = 16;   %number of filters for first convolution layer
filter_c2 = 32;   %number of filters for second convolution layer
filter_c3 = 128;  %number of filters for final convolution layer
lrate = 0.001;    %learning rate
alpha = 0.25;     %Momentum coefficient
stride_1 = 1;         %Stride is set to 1
stride_2 = 2;         %Stride for second and third CNN layer
zeropad = 1;        %Zero Padding = (filterSize - 1)/2    

%Parameters for CNN
wc1_init = normrnd(0,10^-3, filterSize,filterSize,num_channels,filter_c1);
wc2_init = normrnd(0,10^-3, filterSize,filterSize,1,filter_c2);
wc3_init = normrnd(0,10^-3, filterSize,filterSize,1,filter_c3);
bc1_init = normrnd(1,10^-3,1,1,1,filter_c1);
bc2_init = normrnd(1,10^-3,1,1,1,filter_c2);
bc3_init = normrnd(1,10^-3,1,1,1,filter_c3);

%Parameters for fully connected layer after the feature extraction
w_fc_init1 = normrnd(0,0.01,256,filter_c3*4*4);
b_fc_init1 = normrnd(0,0.01,256,1);
w_fc_init2 = normrnd(0,0.01,10,256);
b_fc_init2 = normrnd(0,0.01,10,1);

%Variables to save gradients of the weight for each layer
delta_wc_1 = zeros(size(wc1_init));
delta_wc_2 = zeros(size(wc2_init));
delta_wc_3 = zeros(size(wc3_init));  
delta_bc_1 = zeros(size(bc1_init));
delta_bc_2 = zeros(size(bc2_init));
delta_bc_3 = zeros(size(bc3_init));

%Temporary Parameters for fully connected layer after the feature extraction
delta_w_fc_init_1 = zeros(size(w_fc_init1));
delta_b_fc_init_1 = zeros(size(b_fc_init1));
delta_w_fc_init_2 = zeros(size(w_fc_init2));
delta_b_fc_init_2 = zeros(size(b_fc_init2));
labels = zeros(10,1);
accuracy = 0;
error = 0;

%Start the training process
for i = 1:epoch
    %Random shuffle of the dataset for SGD
    shuffle = randperm(length(labels_train));
    for j = 1:num_batch
        %Save the features and labels of the batch samples in a temporary variable
        tempx = features_train(:,:,:,shuffle((j-1)*samples_batch+1:j*samples_batch));
        tempd = labels_train(shuffle((j-1)*samples_batch+1:j*samples_batch));
        
        %Forward Propogation for first Convolutional layer
        v_c1 = FF_CLayer(tempx, stride_1, zeropad, wc1_init,  bc1_init);
        z_c1 = ReLu(v_c1);  %ReLu layer
        o_c1 = FF_Pool(z_c1,2,2);  %MaxPooling with pooling filter - 2x2
        %Forward Propogation for second Convolutional layer
        v_c2 = FF_CLayer(o_c1, stride_1, zeropad, wc2_init,  bc2_init);
        z_c2 = ReLu(v_c2);  %ReLu layer
        o_c2 = FF_Pool(z_c2,2,stride_2);  %MaxPooling with pooling filter - 2x2
        %Forward Propogation for third Convolutional layer
        v_c3 = FF_CLayer(o_c2, stride_1, zeropad, wc3_init,  bc3_init);
        z_c3 = ReLu(v_c3);  %ReLu layer
        o_c3 = FF_Pool(z_c3,2,stride_2);  %MaxPooling with pooling filter - 2x2
        %Forward propogation of fully connected neurons
        input_fc = reshape(o_c3,size(o_c3,1)*size(o_c3,2)*size(o_c3,3),samples_batch);
        v_fc = w_fc_init1*input_fc + b_fc_init1;
        o_fc = ReLu(v_fc);
        %Without Dropout
        %o_fc = ReLu(v_fc);
        %v_final = w_fc_init2*o_fc + b_fc_init2;
        %output = softmax(v_final);
        %Do dropout on the fully connected layer
        drop_fc = DropOut(o_fc,0.25);    %Dropout with threshold probability of 0.25
        v_final = w_fc_init2*drop_fc + b_fc_init2;
        output = softmax(v_final);
        %The desired results
        d = full(ind2vec(tempd'+1));
        
        %Calculate the gradients
        delta_b_fc_init2 = (1/samples_batch)*sum(output - d,2);
        delta_w_fc_init2 = (1/samples_batch)*(output - d)*o_fc';
        temp = (w_fc_init2'*(output-d));
        %BackPropogation without the dropout
        %temp = (w_fc_init2'*(output-d));
        %delta_Relu_bfc = diff_Relu(temp);
        %delta_b_fc_init1 = (1/samples_batch)*sum(delta_Relu_bfc,2);
        %Dropout Backpropogation
        delta_drop = temp.*drop_fc/0.5;
        delta_Relu_bfc = diff_Relu(delta_drop);
        delta_b_fc_init1 = (1/samples_batch)*sum(delta_Relu_bfc,2);        
        delta_w_fc_init1 = (1/samples_batch)*delta_Relu_bfc*input_fc';
        delta_input = w_fc_init1'*delta_Relu_bfc;
        delta_input = reshape(delta_input,size(o_c3));
        %Now do backward propogation for Convolutional layer - 3
        delta_pool3 = Bprop_Pool(delta_input,z_c3,2);
        delta_Relu3 = diff_Relu(delta_pool3);
        [delta_back2, delta_wc3, delta_bc3] = Bprop_conv(delta_Relu3, o_c2, wc3_init, zeropad);
        %Now do backward propogation for Convolutional layer - 2
        delta_pool2 = Bprop_Pool(delta_back2,z_c2,2);
        delta_Relu2 = diff_Relu(delta_pool2);
        [delta_back1, delta_wc2, delta_bc2] = Bprop_conv(delta_Relu2, o_c1, wc2_init, zeropad);
        %Now do backward propogation for Convolutional layer - 1
        delta_pool1 = Bprop_Pool(delta_back1,z_c1,2);
        delta_Relu1 = diff_Relu(delta_pool1);
        [~, delta_wc1, delta_bc1] = Bprop_conv(delta_Relu1, tempx, wc1_init, zeropad);
        
        %Calculate the total gradients for all the parameters
        delta_wc_1 = alpha*delta_wc_1 + lrate*delta_wc1;
        delta_wc_2 = alpha*delta_wc_2 + lrate*delta_wc2;
        delta_wc_3 = alpha*delta_wc_3 + lrate*delta_wc3;
        delta_bc_1 = alpha*delta_bc_1 + lrate*delta_bc1;
        delta_bc_2 = alpha*delta_bc_2 + lrate*delta_bc2;
        delta_bc_3 = alpha*delta_bc_3 + lrate*delta_bc3;
        delta_w_fc_init_1 = alpha*delta_w_fc_init_1 + lrate*delta_w_fc_init1;
        delta_w_fc_init_2 = alpha*delta_w_fc_init_2 + lrate*delta_w_fc_init2;
        delta_b_fc_init_1 = alpha*delta_b_fc_init_1 + lrate*delta_b_fc_init1;
        delta_b_fc_init_2 = alpha*delta_b_fc_init_2 + lrate*delta_b_fc_init2;
        
        %Update the parameters
        wc1_init = wc1_init - delta_wc_1;
        wc2_init = wc2_init - delta_wc_2;
        wc3_init = wc3_init - delta_wc_3;
        bc1_init = bc1_init - delta_bc_1;
        bc2_init = bc2_init - delta_bc_2;
        bc3_init = bc3_init - delta_bc_3;
        w_fc_init1 = w_fc_init1 - delta_w_fc_init_1;
        b_fc_init1 = b_fc_init1 - delta_b_fc_init_1;
        w_fc_init2 = w_fc_init2 - delta_w_fc_init_2;
        b_fc_init2 = b_fc_init2 - delta_b_fc_init_2;
        
    end
    
    %Initialize the temporary variables back to 0
    delta_wc_1 = zeros(size(delta_wc_1));
    delta_wc_2 = zeros(size(delta_wc_2));
    delta_wc_3 = zeros(size(delta_wc_3));
    delta_bc_1 = zeros(size(delta_bc_1));
    delta_bc_2 = zeros(size(delta_bc_2));
    delta_bc_3 = zeros(size(delta_bc_3));
    delta_w_fc_init_1 = zeros(size(delta_w_fc_init1));
    delta_w_fc_init_2 = zeros(size(delta_w_fc_init2));
    delta_b_fc_init_1 = zeros(size(delta_b_fc_init1));
    delta_b_fc_init_2 = zeros(size(delta_b_fc_init2));
    
    
    for j = 1:length(labels_test)
        
        %Forward Propogation for first Convolutional layer
        v_c1 = FF_CLayer(feature_test, stride_1, zeropad, wc1_init,  bc1_init);
        z_c1 = ReLu(v_c1);  %ReLu layer
        o_c1 = FF_Pool(z_c1,2,2);  %MaxPooling with pooling filter - 2x2
        %Forward Propogation for second Convolutional layer
        v_c2 = FF_CLayer(o_c1, stride_1, zeropad, wc2_init,  bc2_init);
        z_c2 = ReLu(v_c2);  %ReLu layer
        o_c2 = FF_Pool(z_c2,2,stride_2);  %MaxPooling with pooling filter - 2x2
        %Forward Propogation for third Convolutional layer
        v_c3 = FF_CLayer(o_c2, stride_1, zeropad, wc3_init,  bc3_init);
        z_c3 = ReLu(v_c3);  %ReLu layer
        o_c3 = FF_Pool(z_c3,2,stride_2);  %MaxPooling with pooling filter - 2x2
        %Forward propogation of fully connected neurons
        input_fc = reshape(o_c3,size(o_c3,1)*size(o_c3,2)*size(o_c3,3),samples_batch);
        v_fc = w_fc_init1*input_fc + b_fc_init1;
        o_fc = ReLu(v_fc);
        v_final = w_fc_init2*o_fc + b_fc_init2;
        output = softmax(v_final);
        class = labels_test(j) + 1;
        max_index = max(output);
        labels(class) = 1;
        e = labels.*log(output);
        e(isnan(e)) = 0;
        error = error + sum(e);
        if (class == max_index)
            accuracy = accuracy + 1;
        end
        
    end
    
    fprintf('The accuracy on the test set: %.2f \n',accuracy);
    fprintf('The cross entropy error on the test set: %.2f \n',error/(length(labels_test)));
    accuracy = 0;
    error = 0;
    
    
end
    %Final Parameters
    w_c1 = wc1_init;
    w_c2 = wc2_init;
    w_c3 = wc3_init;
    b_c1 = bc1_init;
    b_c2 = bc2_init;
    b_c3 = bc3_init;
    w_fc1 = w_fc_init1;
    w_fc2 = w_fc_init2;
    b_fc1 = b_fc_init1;
    b_fc2 = b_fc_init2;        

end

