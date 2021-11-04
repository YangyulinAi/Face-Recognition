msg = "Face Recognition for CNN";
opts = ["CNN Train Gray Image" "CNN Train Colour Image" "CNN Predictor" "CNN Evaluation"  "Exit"];
choice = menu(msg,opts);
while(true)
    if(choice == 1)
        disp("Training Dataset");
        TrainDatabasePath = uigetdir(strcat(matlabroot,"\.."), "Select Training Set Folder");
        [net,traininfo] = CNN_TrainGrayData(TrainDatabasePath);
        g = msgbox("Train Successfully!","System Infomation", 'help');
        waitfor(g);
        CNN_UI;
        return;
    end
    if(choice == 2)
        disp("Training Dataset");
        TrainDatabasePath = uigetdir(strcat(matlabroot,"\.."), "Select Training Set Folder");
        [net,traininfo] = CNN_TrainRGBData(TrainDatabasePath);
        g = msgbox("Train Successfully!","System Infomation", 'help');
        waitfor(g);
        CNN_UI;
        return;
    end
    if(choice == 3)
        disp("Testing Dataset");
        TestDatabasePath = uigetdir(strcat(matlabroot,"\.."), "Select Test Set Folder");
        [accuracy] = CNN_Test(TestDatabasePath,net);
        msg = strcat("Test Accuracy is: ", num2str(accuracy), "%");
        g = msgbox(msg, "System Infomation", 'help');
        waitfor(g);
        CNN_UI;
        return;
    end
    if(choice == 4)
        disp("Evaluate model");  
        TestDatabasePath = uigetdir(strcat(matlabroot,"\.."), "Select Evaluation Set Folder");
        [acc, overall_precision, overall_recall, f1_score, kappa] = Evaluate(TestDatabasePath,net);
        msg = strcat("Evaluation Accuracy is: ", num2str(acc), "%",...
                     "  Overall Precision is: ", num2str(overall_precision),...
                     "  Overall Recall is: ", num2str(overall_recall),...
                     "  f1 Score is: ", num2str(f1_score),...
                     " Kappa Value is: ", num2str(kappa));
        g = msgbox(msg, "System Infomation", 'help');
        waitfor(g);
        CNN_UI;
        return;
    end
    if(choice == 5)
        clc;
        clear;
        close all;
        return;
    end
end