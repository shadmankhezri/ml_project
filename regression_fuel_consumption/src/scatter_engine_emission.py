import matplotlib.pyplot as plt


def show_scatter_engine_emissions(data):
    
    plt.scatter(data['ENGINESIZE'] , data['CO2EMISSIONS'] , color='blue')

    plt.xlabel('ENGINESIZE')
    plt.ylabel('EMISSIONS')
    plt.title('Plot Engine Size vs the Emission, to see how linear is their relationship is\n')

    plt.savefig(f"graph/relationship_enginesize_emission.png")