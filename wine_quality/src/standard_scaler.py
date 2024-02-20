

from sklearn.preprocessing import StandardScaler




def standard_scaler_train(xx_train):
    sc = StandardScaler()

    X_train = sc.fit_transform(xx_train)
    return X_train



def standard_scaler_test(xx_test):
    sc = StandardScaler()

    X_test = sc.fit_transform(xx_test)
    return X_test
