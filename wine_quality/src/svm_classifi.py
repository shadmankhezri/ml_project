

# from sklearn.svm import SVC
from sklearn import svm


def svm_classifier(X_train , y_train , X_test):
    
    clf = svm.SVC()
    clf.fit(X_train , y_train)

    pred_clf = clf.predict(X_test)

    return pred_clf