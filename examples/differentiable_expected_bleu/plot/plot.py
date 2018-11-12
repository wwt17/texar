#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import math
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#rcParams['font.family'] = 'Times New Roman'
fontsize = 17
font = {
    'family': 'Times New Roman',
    'size': fontsize,
}
matplotlib.rc('font', **font)
from nltk.translate.bleu_score import sentence_bleu
from scipy.stats import pearsonr


def mean(a):
    return sum(a) / len(a)

def main(n):
    data = None
    for i in range(n):
        with open("{}.test.pkl".format(i), "rb") as pickle_file:
            _data = []
            while True:
                try:
                    item = pickle.load(pickle_file)
                    texts, losses = item[:3], item[3:]
                    assert all(map(lambda s: len(s) == 1, texts))
                    _data.append(tuple(map(lambda s: s[0], texts)) + losses)
                except EOFError:
                    break
        if data is None:
            data = [[[] for x in item] for item in _data]
        for item, _item in zip(data, _data):
            for x, _x in zip(item, _item):
                x.append(_x)

    for i, item in enumerate(data):
        for j in range(len(item)):
            if j != 2:
                x = item[j]
                _x = x[0]
                try:
                    assert all(map(lambda e: e == _x, x))
                except AssertionError:
                    print('data[{}][{}] = {}'.format(i, j, x))
                item[j] = _x

    res = []
    for target_sent, bs_sent, sample_sents, debleu_loss, mle_loss in data:
        bs_bleu = sentence_bleu([target_sent], bs_sent)
        sample_bleu = mean([sentence_bleu([target_sent], sample_sent) for sample_sent in sample_sents])
        debleu, mle = map(lambda x: math.exp(-x), (debleu_loss, mle_loss / max(1, len(target_sent))))
        _res = bs_bleu, sample_bleu, debleu, mle
        res.append(_res)
        #print("{}\n{}\n{}".format(' '.join(target_sent), ' '.join(bs_sent), ' '.join(sample_sents[0])))
        #print("{}\t{}\t{}".format(*_res))

    labels = ['beam search BLEU', 'sample mean BLEU', 'DEBLEU', 'likelihood']
    colors = ['saddlebrown', 'mediumblue', 'darkgreen', 'crimson']

    vals = list(zip(*res))

    fig = plt.figure()
    fig.set_size_inches(12, 36)

    tot = (lambda x: x * (x-1) / 2)(len(labels))
    cnt = 0
    pearsonr_file = open('pearsonr', 'w')
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            r, p = pearsonr(vals[i], vals[j])
            print('{:.6f}\t{:.6f}'.format(r, p), file=pearsonr_file)
            cnt += 1
            ax = fig.add_subplot(tot, 2, cnt)
            ax.set_title('', fontsize=fontsize)
            ax.set_xlabel(labels[i], fontsize=fontsize)
            ax.set_ylabel(labels[j], fontsize=fontsize)
            ax.scatter(vals[i], vals[j], s=1./3)
            cnt += 1
            ax = fig.add_subplot(tot, 2, cnt)
            ax.set_title('', fontsize=fontsize)
            points = list(zip(vals[i], vals[j]))
            points.sort()
            X = list(range(len(vals[i])))
            plt.bar(X, [+a for a, b in points], facecolor=colors[i],
                    edgecolor=colors[i], label=labels[i])
            plt.bar(X, [-b for a, b in points], facecolor=colors[j],
                    edgecolor=colors[j], label=labels[j])
            plt.legend(loc='upper left')
    pearsonr_file.close()

    plt.savefig("figure.pdf")
    plt.savefig("figure.png")
    plt.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', type=int)
    args = argparser.parse_args()
    main(**vars(args))
