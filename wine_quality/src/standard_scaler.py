

from sklearn.preprocessing import StandardScaler




def standard_scaler_train(x_train):

    sc = StandardScaler()                # make object for StandardScaler class

    X_train = sc.fit_transform(x_train)  # use fittransform for standarding

    return X_train                       # return standard data



def standard_scaler_test(xx_test):
    
    sc = StandardScaler()

    X_test = sc.fit_transform(xx_test)

    return X_test
