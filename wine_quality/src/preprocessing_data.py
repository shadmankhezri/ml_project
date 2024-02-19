"""
Because there are several different qualities of wine, and we want to have quality wine,
we divide wines into two categories, good and bad, in terms of quality. That is, we consider 
wines that have a quality above 6 as good and those below 6 as bad, and we define good and bad labels for them.
And then we determine a numerical value for good and bad by using Sklearn in such a way that 
bad wine gets 0 and good wine gets 1.
"""


from sklearn.preprocessing import LabelEncoder
import pandas as pd


def prepro_data(data):
    
    bins = (2 , 6.5 , 8)
    group_name = ['bad' , 'good']

    data["quality"] = pd.cut(data["quality"] , bins = bins , labels = group_name)

    data = data["quality"].unique()
    return data


def assign_label(data):
    label_quality = LabelEncoder()

    data["quality"] = label_quality.fit_transform(data["quality"])
    return data