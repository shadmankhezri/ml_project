

from sklearn.metrics import r2_score , mean_absolute_error



def calculate_metrics(test_y , predicted_y):

    mae = mean_absolute_error(test_y , predicted_y)
    
    r2 = r2_score(test_y , predicted_y)

    return mae , r2