import os
import argparse

def load_dic(path):
    dic = {}
    with open(path, 'r') as f:
        for k in f.readlines():
            k = k.strip()
            if len(k)==0: continue
            k = k.split()
            dic[k[0]] = k[1]
    return dic

def read_text(path):
    with open(path, 'r') as f:
        text = f.readlines()
    return text

def write_file(src, trg):
    with open(src, 'r') as f1, open(trg, 'w') as f2:
        for k in f1.readlines(): f2.write(k)

def path_exists(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"Path exists check:{path} is not a valid path.")