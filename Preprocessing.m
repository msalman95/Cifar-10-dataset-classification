function [features_train,features_test,labels_train,labels_test] = Preprocessing(file)

%Normalize the data to be in range of [0 1]
features_train = double(file.traindat)./255;
features_test = double(file.testdat)./255;

% Convert the normalized images back to 32x32 RGB images
features_train = permute(reshape(features_train,50000,32,32,3),[3,2,4,1]);
features_test = permute(reshape(features_test,10000,32,32,3),[3,2,4,1]);

%Convert the labels to double
labels_train = double(file.trainlbl);
labels_test = double(file.testlbl);

end