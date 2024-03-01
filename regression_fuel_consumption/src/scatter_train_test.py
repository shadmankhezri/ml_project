import matplotlib.pyplot as plt


def scatter_train_test(train , test):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(train['ENGINESIZE'] , train['CO2EMISSIONS'] , color='blue')
    ax1.scatter(test['ENGINESIZE'] , test['CO2EMISSIONS'] , color='red')

    plt.xlabel('ENGINESIZE')
    plt.ylabel('EMISSIONS')
    plt.title('Plot Train and Test Engine Size vs the Emission, to see how linear is their relationship is\n')

    plt.savefig(f"graph/train_test_enginesize_emission.png")