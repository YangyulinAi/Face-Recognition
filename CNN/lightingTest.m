load('DarkNet-19_V6.mat');
I = imread("./lightingUnknown/S20/8.pgm");
I = imresize(I,[227,227]);
[YPred,probs] = classify(net,I);
subplot(2,1,1); imshow(I);
label = YPred;
labelCell = cellstr(label);
labelstr = cell2mat(labelCell)
title(string(label) + ", " + num2str(100*max(probs), 3) + "%");

path=['./lightingTrain/',labelstr,'/1.pgm']
J = imread(path);
J = imresize(J,[227,227]);
subplot(2,1,2); imshow(J);

