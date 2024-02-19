
from src.display import display
from src.read_file import read_file
from src.show_info_data import show_info_data
from src.check_null_data import check_null_data
from src.preprocessing_data import prepro_data
from src.preprocessing_data import assign_label
from src.quality_counts import quality_value_counts
from src.show_plot_quality import show_plot_quality


PATH = "data/winequality-red.csv"


def main():


    df_wine = read_file(PATH)
    display(f"\nShow the first data frame of wine :\n\n{df_wine}")

    # display(show_info_data(df_wine))

    # display(check_null_data(df_wine))


    # we divide wines in 2 bins , good and bad
    display(prepro_data(df_wine))

    # now bad wine is 0 and good wine is 1
    display(assign_label(df_wine))


    display(quality_value_counts(df_wine))

    show_plot_quality(df_wine["quality"])


if __name__ == "__main__":
    main()