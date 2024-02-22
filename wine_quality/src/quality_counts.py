# with this func show counts of values


def quality_value_counts(data):
    
    data = data["quality"].value_counts()

    return data


