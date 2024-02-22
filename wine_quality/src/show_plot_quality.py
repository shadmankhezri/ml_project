
# with seaborn and matplotlib , show plot of values , 0 and 1


import seaborn as sns
import matplotlib.pyplot as plt


def show_plot_quality(data):

    sns.countplot(data)

    plt.show()