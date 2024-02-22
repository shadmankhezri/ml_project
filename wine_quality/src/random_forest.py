"""
we use class RandomForestClassifier from sklearn and get train data and x_test data
make rfc object for class RandomForestClassifier and The number of trees in the forest, 200.
and fit train data for make a model and finaly we get test x data to predict method for 
prediction.
"""



from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier(X_train , y_train , X_test):

    rfc = RandomForestClassifier(n_estimators=200)        # make object 

    rfc.fit(X_train, y_train)                       
 
    pred_rfc = rfc.predict(X_test)

    return pred_rfc