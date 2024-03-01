
import matplotlib.pyplot as plt


def show_histogram(data):
    
    plt.hist(data)

    plt.xlabel('ENGINESIZE')
    plt.ylabel('COUNT')
    plt.title('Graphic show of the number of ENGINESIZE\n')

    plt.savefig(f"graph/count_engine_size.png")



# def show_scatter_for_emissions(a , b):
    
#     plt.scatter(a , b , color='blue')

#     plt.xlabel('ENGINESIZE')
#     plt.ylabel('EMISSIONS')
#     plt.title('Plot Engine Size vs the Emission, to see how linear is their relationship is\n')

#     plt.savefig(f"graph/relationship_enginesize_emission.png")