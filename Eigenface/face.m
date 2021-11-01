% Face recognition using eigenfaces
 
close all,  clear  clc;
 

num_trainImg = 5;
showEigenfaces = true;
 

Dirdata = './CroppedYale';
Fildatae = 'test.mat';
if ~exist(datafile, 'file')
	DatasetReader(Dirdata, Fildatae);
end
load(Fildatae); 
 

ClassNum = size(unique(labels), 2);
traindix = [];
testdix = [];
for i=1:ClassNum
	label = find(labels == i);
	divice = randperm(numel(label));
	traindix = [traindix label(divice(1:num_trainImg))];
	testdix = [testdix label(divice(num_trainImg+1:end))];
end


train_x = double(data(:, traindix));
train_y = labels(traindix);
test_x = double(data(:, testdix));
test_y = labels(testdix);


tic;
[num_dim, num_imgs] = size(train_x);   
avg_face = mean(train_x, 2); 			
X = bsxfun(@minus, train_x, avg_face); 
 


M = sqrt(numel(avg_face));
N = numel(avg_face) / M;
imagesc(reshape(avg_face, M, N)); title('Mean face');
colormap('gray');
 
if num_dim <= num_imgs 
	C = X * X';
	[V, D] = eig(C);
else
	C = X' * X; 
	[U, D] = eig(C);
	V = X * U;
end
eigenfaces = V;
eigenfaces = eigenfaces ./ (ones(size(eigenfaces,1),1) * sqrt(sum(eigenfaces.*eigenfaces)));
toc;


X_t = bsxfun(@minus, test_x, avg_face);
T_t = eigenfaces' * X_t;

X = bsxfun(@minus, train_x, avg_face);
T = eigenfaces' * X;

 

figure;
num_eigenfaces_show = 9;
for i = 1:num_eigenfaces_show
	subplot(3, 3, i)
	imagesc(reshape(eigenfaces(:, end-i+1), M, N));
	title(['Eigenfaces ' num2str(i)]);
end
colormap('gray');


AB = -2 * T_t' * T;      
BB = sum(T .* T);         
distance = bsxfun(@plus, AB, BB);       
[score, index] = min(distance, [], 2);   
 

matchCount = 0;
for i=1:numel(index)
	predict = train_y(index(i));
	if predict == test_y(i)
		matchCount = matchCount + 1;
	end
end
fprintf('accuracy: %0.3f%% \n', 100 * matchCount / numel(index));



function DatasetReader(dataDir, saveName)
	dirs = dir(dataDir);
	data = [];
	labels = [];
	for i = 3:numel(dirs)
		imgDir = dirs(i).name;
		imgDir = fullfile(dataDir, imgDir);
		ListImage = dir(fullfile(imgDir, '*.pgm'));
		for j = 1:numel(ListImage)
			Name_img = ListImage(j).name;
			if strcmp('Ambient.pgm',  Name_img(end-10:end))
				continue;
			end
			im = imread(fullfile(imgDir, Name_img));
			if size(im, 3) ==3
				im = rgb2gray(im);
			end
			im = imresize(im, [32 32]);
			im = reshape(im, 32*32, 1);
			data = [data im];
		end
		labels = [labels ones(1, numel(ListImage)-1) * (i-2)];
	end
	save(saveName, 'data', 'labels');
end

