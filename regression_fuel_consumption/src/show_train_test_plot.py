# send the train and test data with model for show in graph
# we use scatter plot for train and test with different color


import matplotlib.pyplot as plt


def show_train_test_plot(train_X , train_y , test_X , test_y , model):
    
    plt.scatter(train_X , train_y , color='blue' , label='Train Data')
    plt.scatter(test_X , test_y , color='red' , label='Test Data')

    plt.plot(train_X , model.predict(train_X) , color='green' , label='Regression Line')
    
    plt.legend()

    plt.savefig(f"graph/simple_regression.png")