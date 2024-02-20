

from sklearn.neural_network import MLPClassifier


def mlp_classifier(X_train , y_train , X_test):

    mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11) , max_iter=500)

    mlpc.fit(X_train , y_train)
    pred_mlpc = mlpc.predict(X_test)

    return pred_mlpc