
import pandas as pd

# make function for read csv file from pandas

def read_file(PATH):
    
    df_fuel = pd.read_csv(PATH)

    return df_fuel