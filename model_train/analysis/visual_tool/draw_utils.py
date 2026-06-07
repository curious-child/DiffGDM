
def list_of_groups(init_list, children_list_len):
    """
    :param init_list: 原列表
    :param children_list_len: 指定切割的子列表的长度（就是你要分成几份）
    :return: 新列表
    """
    list_of_groups = zip(*(iter(init_list),) * children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list