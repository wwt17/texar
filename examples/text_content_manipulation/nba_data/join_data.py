from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

stages = ['train', 'valid', 'test']
ref_strs = ['', '_ref']
fields = ['entry', 'attribute', 'value']

def read_content(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip().split()

if __name__ == '__main__':
    for stage in stages:
        for ref_str in ref_strs:
            contents = [read_content('nba.{}{}.{}.txt'.format(field, ref_str, stage)) for field in fields]
            with open('gold{}.{}.txt'.format(ref_str, stage), 'w') as outfile:
                for paired_lines in zip(*contents):
                    lens = list(map(len, paired_lines))
                    assert all(l == lens[0] for l in lens)
                    print(' '.join(map('|'.join, zip(*paired_lines))), file=outfile)
