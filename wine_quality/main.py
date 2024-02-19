
from src.display import display
from src.read_file import read_file


PATH = "data/winequality-red.csv"

def main():

    df_wine = read_file(PATH)
    display(df_wine)




if __name__ == "__main__":
    main()