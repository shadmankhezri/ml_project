
import matplotlib.pyplot as plt


def show_histogram(data):
    
    plt.hist(data['ENGINESIZE'] , bins=10)

    plt.xlabel('ENGINESIZE')
    plt.ylabel('COUNT')
    plt.title('Graphic show of the number of ENGINESIZE\n')

    plt.savefig(f"graph/count_engine_size.png")
