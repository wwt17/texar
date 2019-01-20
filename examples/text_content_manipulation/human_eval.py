#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
from collections import Counter, namedtuple
from utils import read_sents_from_file, read_x, read_y


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('mode', choices=['score', 'compare'])
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('-n', type=int, default=100)
    argparser.add_argument('--data_prefix', default='nba_data/nba.')
    argparser.add_argument('--stage', default='test')
    argparser.add_argument('files', nargs='+')
    argparser.add_argument('--sent_file', default='examples.sent.txt')
    argparser.add_argument('--res_file', default='human_eval_res.txt')
    args = argparser.parse_args()

    if args.mode == 'score':
        assert len(args.files) == 1
    else:
        assert len(args.files) > 1

    x = read_x(args.data_prefix, 0, args.stage)
    y_ = read_y(args.data_prefix, 1, args.stage)

    list_of_sents = [x, y_] + list(map(read_sents_from_file, args.files))
    lens = list(map(len, list_of_sents))
    print('#examples:', ' '.join(map(str, lens)))
    assert all(l == lens[0] for l in lens), "Not all #examples are equal."
    paired_sents = list(zip(*list_of_sents))
    
    random.seed(args.seed)
    examples = random.sample(paired_sents, args.n)

    res_list = [() for sents in examples]
    if args.mode == 'compare':
        print('0: equally good/bad  1: 1st is better  2: 2nd is better')
    else:
        print('scores are integer between 1 and 5 (from worst to best)')

    def print_sent(sent, name):
        print('{:<2}: {}'.format(name, ' '.join(map(str, sent))))

    i = 0
    while i < len(examples):
        print('example #{}:'.format(i))
        example = examples[i]
        x, y_, sents = example[0], example[1], example[2:]
        perm = list(range(len(sents)))
        random.shuffle(perm)
        print_sent(x, 'x')
        print_sent(y_, "y'")
        for j, idx in enumerate(perm):
            print_sent(sents[idx], j+1)
        while True:
            try:
                res = input('which is better: ' if args.mode == 'compare' else 'fluency and content scores: ')
                if res == 'p':
                    i -= 1
                    break
                if res == 'n':
                    i += 1
                    break
                if res[0] == 'g':
                    i = int(res[1:])
                    break
                res = list(map(int, res.split()))
                break
            except ValueError:
                print('Invalid input. Please input again.')

        if isinstance(res, str):
            continue

        try:
            if args.mode == 'compare':
                if len(res) != 1:
                    print('Invalid.')
                else:
                    if res[0] != 0:
                        res[0] = perm[res[0]-1] + 1
                    res_list[i] = res
                    i += 1
            else:
                if len(res) != 2:
                    print('Invalid.')
                else:
                    res_list[i] = res
                    i += 1
        except:
            print("Invalid.")
            continue

    with open(args.res_file, 'w') as res_f, open(args.sent_file, 'w') as sent_f:
        for example, res in zip(examples, res_list):
            x, y_, sents = example[0], example[1], example[2:]
            print('\t'.join(map(str, res)), file=res_f)

            for sent in sents:
                print(' '.join(sent), file=sent_f)
            print('\t'.join(map(str, res)), file=sent_f)

    counters = list(map(Counter, zip(*res_list)))
    for i, counter in enumerate(counters):
        print('statistics_{}: {}'.format(i, ' '.join('{}: {}'.format(res, cnt) for res, cnt in sorted(counter.items()))))
