

from sklearn.ensemble import RandomForestClassifier






def random_forest_classifier(X_train , y_train , X_test):

    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)

    return pred_rfc