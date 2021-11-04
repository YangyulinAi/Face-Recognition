function [accuracy] = CNN_Test(testDataPath,net)

testData = imageDatastore(testDataPath ,'LabelSource','foldernames','IncludeSubfolders', true);
total = length(testData.Files);
TP = 0;
TN = 0;
for i = 1:total
    img = read(testData);
    img = imresize(img,[227,227]);
    [YPred,probs] = classify(net,img);
    msg = strcat(string(YPred), " ", num2str(100*max(probs), 3) + "%");
    imshow(img);
    hold on;
    set(figure(1),'Name', msg);
    title(strcat("Predicted Class: ", string(YPred), " Probability: ", num2str(100*max(probs), 3) + "%"));
    hold off;
    
    if(YPred==testData.Labels(i))
        TP = TP + 1;
        disp(strcat(msg, "  Correct Number: ", num2str(TP), "/", num2str(total)));
    else
        TN = TN + 1;
        disp(strcat(msg, "  Incorrect"));
    end
    pause(1);
end
accuracy = TP / total * 100;



