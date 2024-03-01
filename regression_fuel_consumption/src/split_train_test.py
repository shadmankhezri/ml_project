# we use the function train_test_split of model_selection for convert the ENGINESIZE, which is our feature(X), 
# as well as the CO2EMISSIONS, which is our result(Y), into two categories, the 80% train and the 20% test, 


from sklearn.model_selection import train_test_split


def split_train_test(data):
    
    X = data[['ENGINESIZE']]
    y = data['CO2EMISSIONS']

    return train_test_split(X , y , test_size=0.2 , random_state=42)

    # We use random_state so that our data does not change in different run