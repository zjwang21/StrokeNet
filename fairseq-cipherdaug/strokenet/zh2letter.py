import os
import json
import time

import argparse
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

from utils import path_exists, read_text, load_dic, write_file

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if (uchar >= u'\u4e00') and (uchar <= u'\u9fa5'):
        return True
    else:
        return False

def zh2letter(dictionary, line):
    char_set = set(list(line))
    newline = line
    for char in char_set:
        if is_chinese(char):
            newline = newline.replace(char, ' '+dictionary.get(char, '')+' ')
    return ' '.join(newline.split())+'\n'

def parserr():
    
    parser = argparse.ArgumentParser(
        prog="zh2letter",
        usage="%(prog)s --input [options]",
        description="Arguments for generating latinized stroke text from an given plaintext.",
    )
    parser.add_argument('--workers', type=int, required=True, help='workers.')
    parser.add_argument('-i', '--input', type=path_exists, required=True, help='input file dir.')
    parser.add_argument('-o', '--output', type=str, required=True, help='output file dir.')
    parser.add_argument('-v', '--vocab-path', type=path_exists, required=True, help='vocab file path.')
    return parser

if __name__ == '__main__':
    parser = parserr()
    args = parser.parse_args()
    dic = load_dic(args.vocab_path)
    func = partial(zh2letter, dic)
    plaintext_src = read_text(args.input)
    print('Generating latinized strokes......')
    t = time.time()
    with Pool(args.workers) as p:
        iter = p.map(func, plaintext_src)
        print('Finished at {}s. Writing......'.format(time.time()-t))
        with open(args.output, 'w') as f:
            for k in tqdm(iter): f.write(k)
    print('done')
