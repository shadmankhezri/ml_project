from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def train_test_x(data):
    X = data.drop('quality', axis = 1)
    return X



def train_test_y(data):
    y = data['quality']
    return y



def x_train(X , y):
    X_train = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train


def x_test(X , y):
    X_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_test


def y_ttrain(X , y):
    y_train = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return y_train


def y_ttest(X , y):
    y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return y_test