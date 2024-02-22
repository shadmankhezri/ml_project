
# for X we drop quality columns and X will be feature var

def train_test_x(data):
    X = data.drop('quality', axis = 1)
    return X


# for y we just use quality columns just for response

def train_test_y(data):
    y = data['quality']
    return y


