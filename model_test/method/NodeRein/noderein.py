
from ctypes import *
import re

import win32api
from networkx import generate_edgelist


def NodeRein(G,remove_list,target):
    path = 'method/NodeRein/graph.txt'
    with open(path, "wb") as fh:
        for line in generate_edgelist(G, ' ', data=False):
            line = "D " + line + "\n"
            fh.write(line.encode("utf-8"))
        for i in remove_list:
            line='S '+str(i)+"\n"
            fh.write(line.encode("utf-8"))

    dll = CDLL("method/NodeRein/NodeRein.dll")
    dll.main(create_string_buffer(path.encode(encoding="utf-8")), c_int(target))
    final_remove_list = []
    with open(path+'_out', 'r') as fo:
        for line in fo:
            if 'S' in line:
                number = re.findall("\d+", line)
                number = int(number.pop())
                final_remove_list.append(number)
    win32api.FreeLibrary(dll._handle)
    return final_remove_list