# use mudole liner_model and class LinearRegression from sklearn for make model
# create object of LinearRegression class that name is model
# and we fit the train data for making model

from sklearn.linear_model import LinearRegression


def train_linear_model(train_X , train_y):

    model = LinearRegression()              # object for class

    model.fit(train_X , train_y)            # create model


    print(f"Coefficients: {model.coef_}")      # teta 1
    print(f"Intercept: {model.intercept_}")    # teta 0
    print(100*"*")
    
    return model