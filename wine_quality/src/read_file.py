"""
From inside the main module, we called this function to read the file and we also gave 
the PATH of the dataframe as input, and here the information of the dataframe is returned.
"""

import pandas as pd

def read_file(PATH):

    df_wine = pd.read_csv(PATH)

    return df_wine      # return all recordes of first dataframe