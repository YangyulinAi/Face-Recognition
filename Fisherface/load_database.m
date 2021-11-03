function Train = load_database(TrainDatabasePath)
% parameter TrainDatabasePath: the path to the selected train database
% return Train: contains all column vectors

% List folder contents
TrainDatabase = dir(TrainDatabasePath);

% Calculate the total number of folders in train database (num of people)
Train_sets = 0;
for i = 1:size(TrainDatabase, 1)
    if not(strcmp(TrainDatabase(i).name, ".") | strcmp(TrainDatabase(i).name, "..") | strcmp(TrainDatabase(i).name, ".DS_Store"))
        Train_sets = Train_sets + 1;
    end
    
end

% Read all photos in train database and each image (2D) is converted to a
% matrix containing many column matrices
Train = [];
for i = 1 : Train_sets 
    str = int2str(i);
    for j = 1:7    % Seven photos for each person
        f = int2str(j);
        % The path to each photo
        lstr = strcat(TrainDatabasePath, "/S", str, "/", f, ".pgm");
        % Read an image from graphics file
        img = imread(lstr);
        [irow, icol] = size(img);
        % Converted to a column matrix (d * 1) 
        re_img = reshape(img', irow*icol, 1);
        % Stored all column matrices (d * n)
        Train = [Train re_img];
    end
end
% Convert single to double
Train = double(Train);


