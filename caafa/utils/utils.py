def get_missing(x):
    """Checks each value of x if None and replaces it with 1 if true and 0 if false
    """
    x = x.copy()

    for i in x:
        if x[i] is None:
            x[i] = 1
        else:
            x[i] = 0
    
    return x

def get_not_missing(x):
    """Checks each value of x if None and replaces it with 0 if true and 1 if false
    """
    x = x.copy()

    for i in x:
        if x[i] is None:
            x[i] = 0
        else:
            x[i] = 1
    
    return x
