"""
from sklearn we use svm module for make svm model and we use train data and x_test data for test model.
make an object "clf" from class SVC and after that fit the data and predict test data
"""

# from sklearn.svm import SVC
from sklearn import svm


def svm_classifier(X_train , y_train , X_test):
    
    clf = svm.SVC()

    clf.fit(X_train , y_train)

    pred_svm = clf.predict(X_test)

    return pred_svm