
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


from src.display import display
from src.read_file import read_file
from src.show_info_data import show_info_data
from src.check_null_data import check_null_data
from src.preprocessing_data import prepro_data
from src.preprocessing_data import assign_label
from src.quality_counts import quality_value_counts
from src.show_plot_quality import show_plot_quality
# from src.rf_classifier import random_forest_classifier

from src.train_test import train_test_x , train_test_y , x_train , x_test , y_ttrain , y_ttest

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


    
    X = train_test_x(df_wine)

    y = train_test_y(df_wine)


    # X = df_wine.drop('quality', axis = 1)
    # y = df_wine['quality']

    x_train(X , y)
    x_test(X , y)
    y_ttrain(X , y)
    y_ttest(X , y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # sc = StandardScaler()

    # X_train = sc.fit_transform(X_train)
    # X_test = sc.fit_transform(X_test)


    # random_forest_classifier(X_train , y_train)



if __name__ == "__main__":
    main()