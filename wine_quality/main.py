
from src.display import display
from src.read_file import read_file
from src.show_info_data import show_info_data
from src.check_null_data import check_null_data
from src.preprocessing_data import prepro_data
from src.preprocessing_data import assign_label
from src.quality_counts import quality_value_counts
from src.show_plot_quality import show_plot_quality
from src.train_test import train_test_x , train_test_y
from src.standard_scaler import standard_scaler_train , standard_scaler_test
from sklearn.model_selection import train_test_split
from src.rf_classifier import random_forest_classifier
from src.classification_report_matrix import classific_report , confus_matrix



PATH = "data/winequality-red.csv"


def main():


    df_wine = read_file(PATH)
    display(f"\nShow the first data frame of wine :\n\n{df_wine}")

    # display(show_info_data(df_wine))

    # display(check_null_data(df_wine))


    # we divide wines in 2 bins , good and bad
    display(prepro_data(df_wine))

    # now bad wine is 0 and good wine is 1
    display(assign_label(df_wine))


    display(quality_value_counts(df_wine))

    # show_plot_quality(df_wine["quality"])


  
    x = train_test_x(df_wine)   # return X

    y = train_test_y(df_wine)   # return y  


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)




    X_train = standard_scaler_train(X_train)   # return standard X_train
    display(X_train)


    X_test = standard_scaler_test(X_test)      # return standard X_test
    display(X_test)



    pred_rfc = random_forest_classifier(X_train , y_train , X_test)
    display(pred_rfc)                                                 # return pred_rfc


    display(classific_report(y_test , pred_rfc))
    display(confus_matrix(y_test , pred_rfc))


if __name__ == "__main__":
    main()