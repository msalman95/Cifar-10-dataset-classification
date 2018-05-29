clc;clear;

%Preprocess the data
file = importdata('final_project_dataset.mat');
[features_train, features_test, labels_train, labels_test] = Preprocessing(file);



%************************************************************************************************
%***************************Convolutional Neural Network ****************************************
%************************************************************************************************
stride_1 = 1;         %Stride is set to 1
stride_2 = 2;         %Stride for second and third CNN layer
zeropad = 1;          %Zero Padding = (filterSize - 1)/2

%Train the data to get the final weights --fc = fully connected
fprintf('*****************************************************************************************\n');
fprintf('Start of Training\n');
fprintf('*****************************************************************************************\n\n');

[w_conv_layer1, b_conv_layer1, w_conv_layer2, b_conv_layer2, w_conv_layer3, b_conv_layer3, w_fc_layer1, ...
 b_fc_layer1 w_fc_layer2 b_fc_layer2] = TrainCNN(features_train, features_test, labels_train, labels_test);

fprintf('*****************************************************************************************\n');
fprintf('End of Training\n');
fprintf('*****************************************************************************************\n\n');

for j = 1:length(labels_test)
        
        %Forward Propogation for first Convolutional layer
        v_c1 = FF_CLayer(feature_test, stride_1, zeropad, w_conv_layer1,  b_conv_layer1);
        z_c1 = ReLu(v_c1);  %ReLu layer
        o_c1 = FF_Pool(z_c1,2,2);  %MaxPooling with pooling filter - 2x2
        %Forward Propogation for second Convolutional layer
        v_c2 = FF_CLayer(o_c1, stride_1, zeropad, w_conv_layer2,  b_conv_layer2);
        z_c2 = ReLu(v_c2);  %ReLu layer
        o_c2 = FF_Pool(z_c2,2,stride_2);  %MaxPooling with pooling filter - 2x2
        %Forward Propogation for third Convolutional layer
        v_c3 = FF_CLayer(o_c2, stride_1, zeropad, w_conv_layer3,  b_conv_layer3);
        z_c3 = ReLu(v_c3);  %ReLu layer
        o_c3 = FF_Pool(z_c3,2,stride_2);  %MaxPooling with pooling filter - 2x2
        %Forward propogation of fully connected neurons
        input_fc = reshape(o_c3,size(o_c3,1)*size(o_c3,2)*size(o_c3,3),samples_batch);
        v_fc = w_fc_layer1*input_fc + b_fc_layer1;
        o_fc = ReLu(v_fc);
        v_final = w_fc_layer2*o_fc + w_fc_layer2;
        output = softmax(v_final);
        class = labels_test(j) + 1;
        max_index = max(output);
        labels(class) = 1;
        e = labels.*log(output);
        e(isnan(e)) = 0;
        error = error + sum(e);
        prediction(j) = class - 1;
        if (class == max_index)
            accuracy = accuracy + 1;
        end
        
end
    
%Print the accuracy and the cross entropy error
fprintf('The accuracy on the test set: %.2f \n',accuracy);
fprintf('The cross entropy error on the test set: %.2f \n',error/(length(labels_test)));

%Save the confusion matrox
conf_matrix = confusionmat(test_labels,prediction);


%%
%Train the grayscale version of the images
load('final_project_dataset.mat');
load('train_grayscale.mat');
load('test_grayscale.mat');

labels_train = double(trainlbl);
labels_test = double(testlbl);
%%
%Initialize the variables
epoch = 50;
Entropy = zeros(epoch,1);
Accuracy = zeros(epoch,1);
learning_rate = 0.01;
alpha = 0.1;
hidden_neurons = 250;
%Use Xaviers initialization for the weights and bias
w_layer1 = -sqrt(6/(1024 + hidden_neurons)) + 2*sqrt(6/(1024 + hidden_neurons))*rand(hidden_neurons,1024);
b_layer1 = -sqrt(6/(1024 + hidden_neurons)) + 2*sqrt(6/(1024 + hidden_neurons))*rand(hidden_neurons,1);
w_layer2 = -sqrt(6/(1024 + hidden_neurons)) + 2*sqrt(6/(1024 + hidden_neurons))*rand(10,hidden_neurons);
b_layer2 = -sqrt(6/(1024 + hidden_neurons)) + 2*sqrt(6/(1024 + hidden_neurons))*rand(10,1);

%Temporary variables
labels = zeros(10,1);
delta_b1 = zeros(size(b_layer1));
delta_w1 = zeros(size(w_layer1));
delta_b2 = zeros(size(b_layer2));
delta_w2 = zeros(size(w_layer2));

for i = 1:epoch
    %Shuffle the data for SGD
    shuffle = randperm(length(labels_train));
 
    %Pass over an epoch
    for l = 1:length(labels_train)
        class = labels_train(shuffle(l));
        labels(class + 1) = 1;        
        %Forward Propogation
        v1 = w_layer1*features_train(shuffle(l),:)' + b_layer1;
        o1 = 1./(1 + exp(-v1));
        total = w_layer2*o1 + b_layer2;
        output = softmax(total);
        
        %Backpropogation to update weights with momentum
        delta_b2 = alpha*delta_b2 + learning_rate*(output - labels);
        delta_w2 = alpha*delta_w2 + learning_rate*(output - labels)*o1';
        delta_b1 = alpha*delta_b1 + (learning_rate*(output - labels)'*w_layer2)'.*o1.*(1-o1);
        delta_w1 = alpha*delta_w1 + ((learning_rate*(output - labels)'*w_layer2)'.*o1.*(1-o1))*features_train(shuffle(l),:);
        
        b_layer2 = b_layer2 - delta_b2;
        w_layer2 = w_layer2 - delta_w2;
        b_layer1 = b_layer1 - delta_b1;
        w_layer1 = w_layer1 - delta_w1;
        
        labels(:) = 0;
    end
    
    accuracy = 0;
    %Test the trained network on validation set and calculate the error
    for l = 1:length(labels_test)
        class = labels_test(l);
        labels(class + 1) = 1;
        
        %Forward Propogation
        v1 = w_layer1*features_test(l,:)' + b_layer1;
        o1 = 1./(1 + exp(-v1));
        total = w_layer2*o1 + b_layer2;
        output = softmax(total);
        
        max_index = find(output == max(output));
        prediction = max_index - 1;
        %Calculate the accuracy
        if (labels_test(l) == prediction)
            accuracy = accuracy + 1;
        end
        
        %Calculate the entropy error
        error = -labels.*log(output);
        error(isnan(error)) = 0;
        Entropy(i) = Entropy(i) + sum(error);
        labels(:) = 0;
    end
    Accuracy(i) = accuracy*100/length(labels_test);
    Entropy(i) = Entropy(i)/length(labels_test);
       
    fprintf('The cross entropy error is: %.3f, Epoch Number: %d\n',Entropy(i),i);
    fprintf('The Accuracy on the test set is: %.2f%%\n',Accuracy(i));
    
    %Initialize the temporary variables back to zero
    labels(:) = 0;
    delta_b1(:,:) = 0;
    delta_w1(:,:) = 0;
    delta_b2(:,:) = 0;
    delta_w2(:,:) = 0;
    
end


%%
%ANN with one and two hidden layers with colored images
load('final_project_dataset.mat');
%Normalization of the images
norm_train = double(traindat)/255;
norm_test = double(testdat)/255;


[a b c d] = OneHiddenLayer_Training(norm_train, double(trainlbl), norm_test, double(testlbl));
TwoHiddenLayer_Training(norm_train, double(trainlbl), norm_test, double(testlbl));


