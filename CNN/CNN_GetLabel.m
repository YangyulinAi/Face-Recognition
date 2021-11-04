function [labelnum] = CNN_GetLabel(image,net)

img = imread(image);
img = imresize(img,[227,227]);
[YPred] = classify(net,img);
label = YPred;
labelCell = cellstr(label);
labelstr = cell2mat(labelCell);
labelnum = labelstr(2:end);
end


