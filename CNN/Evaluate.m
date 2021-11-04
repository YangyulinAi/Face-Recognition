function [acc, overall_precision, overall_recall, f1_score, kappa] = Evaluate(EvaluateDatabasePath,net)
% parameter EvaluateDatabasePath: the path to the selected evaluate database


num = 0;
corr = 0;

% List folder contents
EvaluateDatabase = dir(EvaluateDatabasePath);
Evaluate_images = 0;

% Calculate the total number of evaluate images
for i = 1:size(EvaluateDatabase, 1)
    if not(strcmp(EvaluateDatabase(i).name, ".") | strcmp(EvaluateDatabase(i).name, "..") | strcmp(EvaluateDatabase(i).name, ".DS_Store"))
        Evaluate_images = Evaluate_images + 1;
    end
end

% Calculate the number of classes
Evaluate_Set = Evaluate_images / 3;    % 3 images for each class

% Initialize to get predicted classes
pred = [];
% Initialize to get actual classes
actual = [];

% Loop all images to evaluate
for i = 1:Evaluate_Set
    istr = int2str(i);
    for j = 1:3   % 3 images for each class
        num = num + 1;
        actual(num, 1) = i;
        jstr = int2str(j);
        TestImage = strcat(istr, "-", jstr);
        % Get the path to each image
        test_image = strcat(EvaluateDatabasePath, "/", TestImage, ".pgm");
        % Find the class and name of each image
        [MStr] = CNN_GetLabel(test_image, net);
        Mnum = str2num(MStr);
        pred(num, 1) = Mnum;
        % Compare predicted class and actual class
        if (Mnum == i)  % Correct prediction
            corr = corr + 1;
        else
            disp(i);
        end
    end
end

% Calculate the accuracy: corrected predicted - total number of images
acc = corr/num;

% Get the confusion matrix
cm = confusionmat(actual, pred);
% Transpose the confusion matrix
cmtp = cm';
% Put TP to the diagonal vector
diagonal = diag(cmtp);

% Get (TP + FP) 
sum_of_rows = sum(cmtp, 2);
% Calculate the precision of each class: TP / (TP + FP)
precision = diagonal ./ sum_of_rows;
% Avoid NaN
precision(isnan(precision)) = 0;
% Calculate the overall precision by finding the mean
overall_precision = mean(precision);

% Get (TP + FN)
sum_of_cols = sum(cmtp, 1);
% Calculate the recall of each class: TP / (TP + FN)
recall = diagonal ./ sum_of_cols';
% Avoid NaN
recall(isnan(recall)) = 0;
% Calculate the overall recall by finding the mean
overall_recall = mean(recall);

% Calculate f1_score
f1_score = 2 * ((overall_precision * overall_recall)/(overall_precision + overall_recall));

% Calculate Kappa
diag_sum = sum(diagonal);
total = sum(cmtp, "all");
p0 = diag_sum / total;
[irow, ~] = size(diagonal);
sum_all_diag = 0;
for i = 1:irow
   sum_diag = sum(cmtp(:, i))*sum(cmtp(i, :));
   sum_all_diag = sum_all_diag + sum_diag;
end
pe = sum_all_diag / (total*total);
kappa = (p0 - pe)/(1-pe);


