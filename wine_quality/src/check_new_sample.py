"""
now after making models we can decide for select one of them and we select 
random forest because the accuracy of it is more than another models. and we use 
random forest model for check the new sample
use a StandardScaler class and RandomForestClassifier class. we get new data of the wine and 
with StandardScaler() should be a standard data for performance.

"""


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def new_wine(new):

    sc = StandardScaler()

    X_new = sc.fit_transform(new)

    return X_new               # return standard data for new wine 




def predect_quality(X_train , y_train , X_new):       # give train data and new wine data for predict quality

    rfc = RandomForestClassifier(n_estimators=200)
    
    rfc.fit(X_train, y_train)
    
    y_new = rfc.predict(X_new)

    return y_new