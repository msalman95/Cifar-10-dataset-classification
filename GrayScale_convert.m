load('final_project_dataset.mat');
%Normalization of the images

%Variables to store data extracted from RGB images
grayscale_data = zeros(32,32,50000);
data_vector = zeros(1024,50000);
grayscale_centralized = zeros(size(grayscale_data));
features_train =  zeros(50000,1024);

% Preprocessing of the data
for i = 1:length(trainlbl)
    colored_image = reshape(traindat(i,:),32,32,3);      %Save each colored image
    %Convert the colored image into grayscale using luminosity model
    grayscale_data(:,:,i) = 0.2126*colored_image(:,:,1) + 0.7152*colored_image(:,:,2) + 0.0722*colored_image(:,:,3);
    %Remove mean of each image from the corresponding grayscaled image
    grayscale_centralized(:,:,i) = grayscale_data(:,:,i) - mean2(grayscale_data(:,:,i));  % subracting the mean
end

std_allpixels = std2(grayscale_data);%standard deviation across all pixels in the whole dataset


for i = 1:length(trainlbl)
    temp = grayscale_centralized(:,:,i);
    temp = temp(:);  %Convert into a vector
    %Check for outliers and quantize them to (+-)3*std
    temp(temp >= 3*std_allpixels) = 3*std_allpixels;
    temp(temp <= -3*std_allpixels) = -3*std_allpixels;    
    temp = (temp - min(temp))/(max(temp) - min(temp));;
    data_vector(:,i) = temp;
    features_train(i,:) = temp;
end

