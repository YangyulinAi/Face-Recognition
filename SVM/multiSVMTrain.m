function multiSVMStruct = multiSVMTrain(TrainData, nSampPerClass, nClass)
%TrainData- Each row is a sample face 
%nSampPerClass- Record the number of samples in each category
%nClass- Number of categories
C = 100;
kertype = 'linear';

% Calculate the classification hyperplane between each two categories
for ii=1:(nClass-1)
    for jj=(ii+1):nClass
        clear X;
        clear Y;
        startPosII = sum( nSampPerClass(1:ii-1) ) + 1;
        endPosII = startPosII + nSampPerClass(ii) - 1;
        X(1:nSampPerClass(ii), :) = TrainData(startPosII:endPosII, :);

        startPosJJ = sum( nSampPerClass(1:jj-1) ) + 1;
        endPosJJ = startPosJJ + nSampPerClass(jj) - 1;
        X(nSampPerClass(ii)+1:nSampPerClass(ii)+nSampPerClass(jj), :) = TrainData(startPosJJ:endPosJJ, :);


        % Set class labels for two-by-two classification
        Y = ones(nSampPerClass(ii) + nSampPerClass(jj), 1);
        Y(nSampPerClass(ii)+1:nSampPerClass(ii)+nSampPerClass(jj)) = -1;

        % classification for the two-by-two classification of individual ii and individual jj
        CASVMStruct{ii,jj}= svmTrain(X',Y',kertype,C);
     end
end

% already classification result
multiSVMStruct.nClass = nClass;
multiSVMStruct.CASVMStruct = CASVMStruct;