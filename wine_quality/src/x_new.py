

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier


def new_wine(new):
    sc = StandardScaler()

    X_new = sc.fit_transform(new)
    return X_new




def y_new(X_train , y_train , X_new):

    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    
    y_new = rfc.predict(X_new)

    return y_new