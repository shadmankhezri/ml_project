

import pandas as pd

def read_file(PATH):
    df_wine = pd.read_csv(PATH)

    return df_wine