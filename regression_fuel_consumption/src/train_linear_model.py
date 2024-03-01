

from sklearn.linear_model import LinearRegression


def train_linear_model(train_X , train_y):

    model = LinearRegression()

    model.fit(train_X , train_y)

    print(f"Coefficients: {model.coef_}")      # teta 1
    print(f"Intercept: {model.intercept_}")    # teta 0
    print(100*"*")
    
    return model