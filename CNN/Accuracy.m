function [accuracy] = Accuracy(testDataPath)
load('DarkNet-19.mat');
testData = imageDatastore(testDataPath ,'LabelSource','foldernames','IncludeSubfolders', true);
Total = length(testData.Files);
TP = 0;
TN = 0;
for i = 1:Total
    im = read(testData);
    im = imresize(im,[227,227]);
    [YPred,probs] = classify(trainedNetwork_1,im);
    if(YPred==testData.Labels(i))
        TP = TP+1;
    else
        TN = TN + 1;      
    end
end
accuracy = TP / Total * 100;
Error = TN / Total * 100;


