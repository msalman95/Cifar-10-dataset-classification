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