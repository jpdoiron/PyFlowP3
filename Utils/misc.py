dict = {}
def get_next_item_name(type : str, separator="_"):
    if type in dict:
        val = dict[type] + 1
        dict[type] = val

    else:
        dict[type] = 0
        val = 0

    res = "{}{}{}".format(type, separator, val)
    return res
