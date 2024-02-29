
import matplotlib.pyplot as plt


def show_histogram(data , x_label):
    
    plt.hist(data)

    plt.xlabel(x_label)
    # plt.xlabel('ENGINESIZE')
    plt.ylabel('COUNT')

    plt.savefig(f"data/histogram_{data}.png")

