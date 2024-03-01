import matplotlib.pyplot as plt


def show_scatter_for_emissions(a , b , c=None):
    
    plt.scatter(a , b , color='blue')

    plt.xlabel('ENGINESIZE')
    plt.ylabel('EMISSIONS')
    plt.title('Plot Engine Size vs the Emission, to see how linear is their relationship is\n')

    plt.savefig(f"graph/{c}_relationship_enginesize_emission.png")