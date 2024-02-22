
from src.display import display
from src.read_file import read_file
from src.show_info_data import show_info_data
from src.change_quality_value import change_quality_value
from src.change_quality_value import change_type_quality_value
from src.quality_counts import quality_value_counts
from src.show_plot_quality import show_plot_quality
from src.train_test import train_test_x , train_test_y
from src.standard_scaler import standard_scaler_train , standard_scaler_test
from sklearn.model_selection import train_test_split
from src.random_forest import random_forest_classifier
from src.report_matrix import classific_report , confus_matrix
from src.svm_model import svm_classifier
from src.mlp_classifier import mlp_classifier
from src.accuracy import accuracy
from src.check_new_sample import new_wine , predect_quality

#-----------------------------------------------
# it is path of the first dataframe
PATH = "data/winequality-red.csv"          

#-----------------------------------------------

def main():

    # call read_file() function for reading the dataframe

    df_wine = read_file(PATH)               
    display(f"\nShow the first data frame of wine :\n\n{df_wine}")

#-----------------------------------------------
    # get the information about dataframe
    
    display(show_info_data(df_wine))                   

#-----------------------------------------------
    # we divide wines in 2 bins , good and bad

    display(change_quality_value(df_wine))              

#-----------------------------------------------
    # now bad wine is 0 and good wine is 1

    df_wine = change_type_quality_value(df_wine)
    display(f"dataframe with number 0 and 1 for quality:\n\n{df_wine}")

#-----------------------------------------------
    # now we count of quality value

    display(quality_value_counts(df_wine))

#-----------------------------------------------
    # show figure of quality values

    show_plot_quality(df_wine["quality"])

#-----------------------------------------------

    # Now seperate the dataset as response variable and feature variabes
  
   
    x = train_test_x(df_wine)   # return X , feature var
    y = train_test_y(df_wine)   # return y , response var
    display(x)
    display(y)

#-----------------------------------------------
    # create train and test with use SKlearn and train_test_split
    # use 20% of data for test and 80% for train and use the random data

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


#-----------------------------------------------
    # Applying Standard scaling to get optimized result

    X_train = standard_scaler_train(X_train)   
    display(f"x_train standard data:\n\n{X_train}")                            # return standard X_train


    X_test = standard_scaler_test(X_test)      # return standard X_test
    display(f"x_test standard data:\n\n{X_test}")

#==================================================================
    
    # the first our model
    # Random Forest Classifier

    pred_rfc = random_forest_classifier(X_train , y_train , X_test)
    display(f"prediction y(quality of test data) of random forest:\n\n{pred_rfc}")                                                 # return pred_rfc

#-----------------------------------------------
    # for see model performed

    display(classific_report(y_test , pred_rfc))
    display(confus_matrix(y_test , pred_rfc))

#-----------------------------------------------
    
    # calculate accuracy for random forest model

    acc_forest_model = accuracy(y_test , pred_rfc)
    display(f"Random forest model gives the accuracy of :{acc_forest_model}")

#==================================================================
    # the second our model 
    # SVM

    # pred_svm = svm_classifier(X_train , y_train , X_test)
    # display(f"prediction y(quality of test data) of SVM:\n\n{pred_svm}")                                                   # return pred_clf

#-----------------------------------------------
    # for see model performed

    # display(classific_report(y_test , pred_svm))
    # display(confus_matrix(y_test , pred_svm))

#-----------------------------------------------
    # calculate accuracy for SVM model

    # acc_svm = accuracy(y_test , pred_svm)
    # display(f"SVM model gives the accuracy of :{acc_svm}")


#==================================================================
    # the third our model
    # neural_network model

    # pred_mlpc = mlp_classifier(X_train , y_train , X_test)
    # display(f"prediction y(quality of test data) of neural_network:\n\n{pred_mlpc}")                                                # return pred_mlpc

#-----------------------------------------------
    # for see model performed

    # display(classific_report(y_test , pred_mlpc))
    # display(confus_matrix(y_test , pred_mlpc))   

#-----------------------------------------------
    
    # acc_neural_network = accuracy(y_test , pred_mlpc)
    # display(f"neural network model gives the accuracy of :{acc_neural_network}")


#==================================================================
    # give new parameters of wine without quality

    new = [[7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]]
    X_new = new_wine(new)
    display(X_new)

#-----------------------------------------------
    # predict quality of new wine

    new_wine_quality = predect_quality(X_train , y_train , X_new)
    display(f"The quality of wine with given parameters is:{new_wine_quality}")
    
#-----------------------------------------------

if __name__ == "__main__":
    main()