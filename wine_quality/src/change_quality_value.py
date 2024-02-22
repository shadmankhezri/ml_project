"""
Because there are several different qualities of wine, and we want to have good quality wine,
we divide wines into two categories, good and bad, in terms of quality. That is, we consider 
wines that have a quality above 6 as good and those below 6 as bad, and we define good and bad
labels for them. And then we determine a numerical value for good and bad by using Sklearn in 
such a way that bad wine gets 0 and good wine gets 1.

"""


from sklearn.preprocessing import LabelEncoder
import pandas as pd


def change_quality_value(data):

    bins = (2 , 6.1 , 8)              # make our bins, we want 2 bins 
    group_name = ['bad' , 'good']     # define names for bins


    data["quality"] = pd.cut(data["quality"] , bins = bins , labels = group_name)


    unique_value = data["quality"].unique()
    return unique_value

#-------------------------------------------

# with labelencoder change bad and good label to 0 and 1 because we want int value for models

def change_type_quality_value(data):

    label_quality = LabelEncoder()

    data["quality"] = label_quality.fit_transform(data["quality"])
    return data