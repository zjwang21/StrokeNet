import os
import json
import time

import argparse
from tqdm import tqdm
from functools import partial
from multiprocessing import set_start_method, Pool

from utils import path_exists, read_text, write_file

def shift_vocab(vocab, key):
    dic = {}
    for i in range(len(vocab)):
        dic[vocab[i]] = vocab[(i+key) % len(vocab)]
    return dic

def monophonic(vocab, shifted_vocab, plain_text):
    cipher_text = []
    for c in plain_text:
        if c in vocab:
            cipher_text.append(shifted_vocab[c])
        else:
            cipher_text.append(c)
    return ''.join(cipher_text)

def parserr():
    
    parser = argparse.ArgumentParser(
        prog="encipher",
        usage="%(prog)s --input --keys [options]",
        description="Arguments for generating cipher text from an given plaintext.",
    )

    parser.add_argument('-i', '--input', type=path_exists, required=True, help='input file dir.')
    parser.add_argument('-s', '--src', type=str, required=True, help='src language.')
    parser.add_argument('-t', '--trg', type=str, required=True, help='trg language.')
    parser.add_argument('--workers', type=int, required=True, help='workers.')
    parser.add_argument('--keys', nargs="+", type=int, required=True, help='list of keys for encipherment.')
    return parser

set_start_method('forkserver', force=True)
if __name__ == "__main__":
    parser = parserr()
    args = parser.parse_args()
    splits = ['train', 'valid']

    vocab = 'etaoinshrdlcumwfgypbvkjxqz'
    for split in splits:
        src_name = "{}.{}-{}.{}".format(split, args.src, args.trg, args.src)
        trg_name = "{}.{}-{}.{}".format(split, args.src, args.trg, args.trg)
        src_file = os.path.join(args.input, src_name)
        trg_file = os.path.join(args.input, trg_name)
        text = read_text(src_file)
        for key in args.keys:
            shifted_vocab = shift_vocab(vocab, key)
            func = partial(monophonic, vocab, shifted_vocab)
            print('Generating ciphered-text of {} data with key {}.'.format(split, key))
            write_file(trg_file, os.path.join(args.input, "{}.{}{}-{}.{}".format(split, args.src, key, args.trg, args.trg)))
            save_src_cipher = os.path.join(args.input, "{}.{}{}-{}.{}{}".format(split, args.src, key, args.trg, args.src, key))
            
            t = time.time()
            pool = Pool(args.workers)
            iter = pool.map(func, text)
            pool.close()
            pool.join()
            print('Finished at {}s. Writing......'.format(time.time()-t))
            with open(save_src_cipher, 'w') as f:
                for k in tqdm(iter): f.write(k)
            

            write_file(src_file, os.path.join(args.input, "{}.{}{}-{}.{}".format(split, args.src, key, args.src, args.src)))
            write_file(save_src_cipher, os.path.join(args.input, "{}.{}{}-{}.{}{}".format(split, args.src, key, args.src, args.src, key)))
            print('done \n')
            del iter
        del text
