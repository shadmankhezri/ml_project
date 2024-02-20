

from sklearn.metrics import confusion_matrix, classification_report



def classific_report(y_test , pred):
    return classification_report(y_test , pred)




def confus_matrix(y_test , pred):
    return confusion_matrix(y_test , pred)