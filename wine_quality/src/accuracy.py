# from metrics module use accuracy function for calculate the accuracy models

from sklearn.metrics import accuracy_score

def accuracy(y_test , pred):

    cm = accuracy_score(y_test, pred)

    return cm