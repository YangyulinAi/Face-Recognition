clc;
clear;
close all;

%% loading the dataset
TrainX = [];
Tr_gnd = [];
nPerson=40;
for i = 1:nPerson
    for j = 1:7
        path = strcat('C:\Users\shi\Desktop\Evaluate\Evaluate\Position + Expression Train\s',num2str(i),'\',num2str(j),'.pgm');
        I = imread(path);
        I = double(I);
        I =imresize(I,[32 32]);
        [m,n] = size(I);
        I = reshape(I,[1 m*n]);
        TrainX = [TrainX;I];
        Tr_gnd = [Tr_gnd;i];
    end
end

TestX = [];
Te_gnd = [];
for i = 1:nPerson
    for j = 1:3
        path = strcat('C:\Users\shi\Desktop\Evaluate\Evaluate\Position + Expression Test\',num2str(i),'-',num2str(j),'.pgm');
        I = imread(path);
        I = double(I);
        I =imresize(I,[32 32]);
        [m,n] = size(I);
        I = reshape(I,[1 m*n]);
        TestX = [TestX;I];
        Te_gnd = [Te_gnd;i];
    end
end

%% PCA feature extraction
nFaces=size(TrainX,1);
[pcaFaces, V] = PCA(TrainX, 30);
[scalingX,A0,B0] = scaling(pcaFaces);
TrainData =scalingX;

for iPerson = 1:nPerson
    nSplPerClass(iPerson) = sum( (Tr_gnd == iPerson) );
end

%% training
multiSVMStruct = multiSVMTrain(TrainData, nSplPerClass, nPerson);
[m n] = size(TestX);
meanVec=mean(TestX);
TestFace = (TestX-repmat(meanVec,m,1))*V; % Dimension reduction
TestFace = scaling(TestFace,1,A0,B0);

%% Classification
predictlabel = multiSVMClassify(TestFace,multiSVMStruct);
nError = sum(predictlabel ~= Te_gnd);
% Performance evaluation calculation method
acc = 1 - nError/length(Te_gnd);
rec_mat= [];  
pre_mat = [];
F1_mat = [];
for j = 1:length(unique(Te_gnd))
    tp = 0;
    fn = 0;
    fp = 0;
    tn = 0;
    for y = 1:m
        if predictlabel(y)==j && Te_gnd(y)==j
            tp=tp+1;
        elseif predictlabel(y)==j && Te_gnd(y)~=j
            fp=fp+1;
        elseif predictlabel(y)~=j && Te_gnd(y)==j
            fn=fn+1;
        elseif predictlabel(y)~=j && Te_gnd(y)~=j
            tn=tn+1;
        end
    end
    rec_tmp= tp/(tp+fn);
    pre_tmp = tp/(tp+fp);
    F1_tmp = 2*(pre_tmp*rec_tmp)/(pre_tmp+rec_tmp);
    rec_mat= [rec_mat rec_tmp];
    pre_mat = [pre_mat pre_tmp];
    F1_mat = [F1_mat F1_tmp];
end
rec_mat(isnan(rec_mat))=0;
pre_mat(isnan(pre_mat))=0;
F1_mat(isnan(F1_mat))=0;
acc
rec = mean(rec_mat)
pre = mean(pre_mat)
F1 = mean(F1_mat)
