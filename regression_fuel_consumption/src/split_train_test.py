


from sklearn.model_selection import train_test_split


def split_train_test(data):
    
    X = data[['ENGINESIZE']]
    y = data['CO2EMISSIONS']

    return train_test_split(X , y , test_size=0.2 , random_state=42)

