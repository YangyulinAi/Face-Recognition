function [MStr, Img_str] = Test(test_image, mean_database, V_PCA, V_Fisher, ProjectedF_Images)
% parameter test_image: string -- the path to the test image
%           mean_database: a column vector -- average face
%           V_PCA: a matrix -- PCA subspace
%           V_Fisher: a matrix -- FDA subspace
%           ProjectedF_Images: data are stored after two projections
% return MStr: string -- the class of the test image
%        Img_str: string -- the match image name

% Find the number of train images in the Fisherface model
num_t_images = size(ProjectedF_Images, 2);

% Read the test image
test_img = imread(test_image);

% Covert the test image (2D) to a column vector
[irow, icol] = size(test_img);
re = reshape(test_img', irow*icol, 1);
re = double(re);

% Calculate the difference between the test column vector and the average
% face
a = re - mean_database;

% Project data to the PCA subspace and the FDA subspace
projected_test = V_Fisher' * V_PCA' * a;

% Calculate the Euclidian distances between the projected test data and every projected train data 
dist = [];
for i = 1:num_t_images
    dist(:, i) = norm(projected_test - ProjectedF_Images(:, i))^2;
end

% Find the index with the smallest distance
[smallest_dist, smallest_num] = min(dist);

% Calculate the class number and the match image name
M = fix(smallest_num / 7);   % 7 images for each class in the train set
mo = mod(smallest_num, 7);   
if(mo == 0)
   m = 7;
else
    m = mo;
    M = M + 1;
end
% Convert integer to string
MStr = int2str(M);
Img_str = int2str(m);


