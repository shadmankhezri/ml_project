"""
we use metrics module from sklearn and use confusion_matrix, classification_report functions
for see how our model performed and Confusion matrix for the random forest classification
"""



from sklearn.metrics import confusion_matrix, classification_report



def classific_report(y_test , pred_rfc):

    return classification_report(y_test , pred_rfc)




def confus_matrix(y_test , pred_rfc):

    return confusion_matrix(y_test , pred_rfc)