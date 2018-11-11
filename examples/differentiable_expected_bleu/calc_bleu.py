#!/usr/bin/env python3

import sys
from nltk.translate.bleu_score import corpus_bleu

if __name__ == '__main__':
    lines = list(map(lambda line: line.strip().split(), sys.stdin.readlines()))
    ref_hypo_pairs = [(lines[i], lines[i+1]) for i in range(0, len(lines), 2)]
    refs, hypos = zip(*ref_hypo_pairs)
    print(corpus_bleu(list(map(lambda ref: [ref], refs)), hypos))
