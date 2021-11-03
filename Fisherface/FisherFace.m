function [mean_database, V_PCA, V_Fisher, ProjectedF_Images] = FisherFace(Train)
% parameter Train: contains all columns matrices (one column for each image)
% return mean_database: a column vector -- average face
%        V_PCA: a matrix -- PCA subspace
%        V_Fisher: a matrix -- FDA subspace
%        ProjectedF_Images: data are stored after two projections

% Get the number of training images
num_images = size(Train, 2);

% Calculate the average face
mean_database = mean(Train, 2);
% Calculate the difference between each face and the average
A = Train - repmat(mean_database, 1, num_images);

% Calculate the covariance matrix for A'*A
C = A' * A;
% Get eigenvetors and eigenvalues for A'*A
[V, D] = eig(C);
% Select 50 eigenvectors with the maximum eigenvalues
V = fliplr(V);
V_selected = [];
for i = 1:50
    V_selected(:, i) = V(:, i); 
end
% Calculate eigenvectors for A*A'
V_PCA = A * V_selected;

% Project each column data to the PCA subspace
ProjectedP_Images = [];
for i = 1:num_images
    ProjectedP_Images(:, i) = V_PCA' * A(:, i);
end

% Calculate the number of classes(num of people)
num_class = num_images / 7;
% Each class has seven images
num_img_per_class = 7;

% Calculate average of all projected data
m_PCA = mean(ProjectedP_Images, 2);
m_class = zeros(50, num_class);

% Initialize to get distances within class
Sw = zeros(50, 50);
% Initialize to get distances between classes
Sb = zeros(50, 50);

for i = 1:num_class
    % Calculate the mean of each class
    m_class(:, i) = mean((ProjectedP_Images(:, ((i-1)*num_img_per_class + 1) : i*num_img_per_class)), 2)';
    
    S = zeros(50, 50);
    for j = ((i-1)*num_img_per_class + 1):(i*num_img_per_class)
        % Calculate the distance between each projected image and the mean
        % of each class
        S = S + (ProjectedP_Images(:, j)-m_class(:, i)) * (ProjectedP_Images(:, j)-m_class(:, i))';
    end
    
    Sw = Sw + S;
    % Get distances between the mean of each class and the mean of all
    % project data
    Sb = Sb + (m_class(:, i) - m_PCA) * (m_class(:, i) - m_PCA)';
end 

% Calcaluate the eigenvectors and eigenvalues of inv(Sw)*Sb
[FV, FD] = eig(Sb, Sw);
FV = fliplr(FV);
V_Fisher = [];
% Ignore the eigenvector with the smallest eigenvalue and create the FDA
% subspace
for i = 1 : num_class-1
    V_Fisher(:, i) = FV(:, i);
end

% Project data to the FDA subspace
ProjectedF_Images = [];
for i = 1: num_images
    ProjectedF_Images(:, i) = V_Fisher' * ProjectedP_Images(:, i);
end




