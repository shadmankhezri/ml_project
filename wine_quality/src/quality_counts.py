# with this func show counts of value and show plot of quality value




def quality_value_counts(data):
    
    data = data["quality"].value_counts()

    return data


