function class = multiSVMClassify(TestFace, multiSVMStruct)

nClass = multiSVMStruct.nClass; % Number of read-in categories
CASVMStruct = multiSVMStruct.CASVMStruct; % Read in the classifier information between the two classes
kertype='linear';


m = size(TestFace, 1);
Voting = zeros(m, nClass); % m test samples, nPerson per sample Ballot box for each category

for iIndex = 1:nClass-1
    for jIndex = iIndex+1:nClass
        classes =svmTest(CASVMStruct{iIndex,jIndex},TestFace,kertype);

        Voting(:, iIndex) = Voting(:, iIndex) + (classes' == 1);
        Voting(:, jIndex) = Voting(:, jIndex) + (classes' == -1);

    end % for jClass
end % for iClass



% final decision by voting result
[vecMaxVal, class] = max( Voting, [], 2 );
end