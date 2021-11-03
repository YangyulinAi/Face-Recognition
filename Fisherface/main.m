% Create the Menu
msg = "Face Recognition for Fisherface";
opts = ["Train Fisherface" "Test on Photos" "Evaluate", "Exist"];
choice = menu(msg,opts); 
while(true)
    if(choice == 1)    % Train Fisherface
        % Open folder selection dialog box (specify the start path in
        % desktop) for the path to train database 
        TrainDatabasePath = uigetdir(strcat(matlabroot,"\.."), "Select Training Set Folder");
        % Load train photos
        Train = load_database(TrainDatabasePath);
        % Train Fisherface Model
        [mean_database, V_PCA, V_Fisher, ProjectedF_Images] = FisherFace(Train);
        % After training, success dialog box displays
        g = msgbox("Train Successfully!!");
        main;
        return;
    end
    if(choice == 2)
        % Check whether the system has already trained the Fisherface
        a = exist("mean_database", "var");
        if (a == 0)
            k = msgbox("Please Train Fisherface First!", "Error", "error");
            main;
            return;
        end
        % Open folder selection dialog box for the path to test database
        TestDatabasePath = uigetdir(strcat(matlabroot,"\.."), "Select Test Set Folder");
        % Create the dialog box to gather the name of the test image
        prompt = ("Enter test image: ");
        dlg_title = "Input of the FisherFace System";
        num_lines = 1;
        def = "1-1";
        TestImage = inputdlg(prompt, dlg_title, num_lines, def);
        % Create the path to the test image
        test_image = strcat(TestDatabasePath, "/", char(TestImage), ".pgm");
        % Read the test image
        test_i = imread(test_image);
        disp(TestImage);
        subplot(121);
        % Display the test image
        imshow(test_i);
        % Find the class of the test image and the match image name
        [MStr, Img_str] = Test(test_image, mean_database, V_PCA, V_Fisher, ProjectedF_Images);
        % Create the path to the match image in the train set
        MatchImage = strcat(TrainDatabasePath, "/S", MStr, "/", Img_str, ".pgm");
        % Read the match image
        MatchImage = imread(MatchImage);    
        subplot(122);
        % Display the match image
        imshow(MatchImage);
        In_File = strcat("Recognition Completed: Photo In S", MStr);
        title(In_File, "FontWeight", "bold", "Fontsize", 16, "color", "blue");
        main;
        return;
    end
    if(choice == 3)
        % Open folder selection dialog box for the path to evaluate
        % database
        EvaluateDatabasePath = uigetdir(strcat(matlabroot,"\.."), "Select Evaluate Set Folder");
        % Evaluate the model and get evaluation results
        [acc, precision, recall, f1_score]= Evaluate(EvaluateDatabasePath, mean_database, V_PCA, V_Fisher, ProjectedF_Images);
        % Convert number to string
        acc = num2str(acc);
        precision = num2str(precision);
        recall = num2str(recall);
        f1_score = num2str(f1_score);
        % Show evalution results
        dis = strcat("Accuracy is ", acc, "; Precision is ", precision, "; Recall is ", recall, "; F1_Score is ", f1_score);
        z = msgbox(dis, "Evaluation Result");
        main;
        return;
    end
    if(choice == 4) 
        % Clear data and exist
        clc;
        clear;
        close all;
        return;
    end

    
end

