

def train_test_x(data):
    X = data.drop('quality', axis = 1)
    return X



def train_test_y(data):
    y = data['quality']
    return y


