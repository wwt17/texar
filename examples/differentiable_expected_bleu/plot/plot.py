#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import math
import pickle
import matplotlib
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

plt.switch_backend('agg')

def mean(a):
    return sum(a) / len(a)

def main():
    data = None
    for i in range(5):
        with open("{}.test.pkl".format(i), "rb") as pickle_file:
            _data = []
            while True:
                try:
                    x, y, z, w = pickle.load(pickle_file)
                    assert len(x) == len(y) == len(z) == 1
                    _data.append((x[0], y[0], z[0], w))
                except EOFError:
                    break
        if data is None:
            data = [(x, y, [], w) for x, y, z, w in _data]
        for A, B in zip(data, _data):
            A[2].append(B[2])
    res = []
    for target_sent, bs_sent, sample_sents, loss_debleu in data:
        bs_bleu = sentence_bleu([target_sent], bs_sent)
        sample_bleu = mean([sentence_bleu([target_sent], sample_sent) for sample_sent in sample_sents])
        _res = bs_bleu, sample_bleu, math.exp(-loss_debleu)
        res.append(_res)
        #print("{}\n{}\n{}".format(' '.join(target_sent), ' '.join(bs_sent), ' '.join(sample_sents[0])))
        #print("{}\t{}\t{}".format(*_res))

    labels = ['beam search BLEU', 'sample mean BLEU', 'DEBLEU']
    colors = ['black', 'blue', 'green']

    vals = list(zip(*res))

    fig = plt.figure()
    fig.set_size_inches(10, 15)

    cnt = 0
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            cnt += 1
            ax = fig.add_subplot(320+cnt)
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.scatter(vals[i], vals[j], s=1./3)
            cnt += 1
            ax = fig.add_subplot(320+cnt)
            points = list(zip(vals[i], vals[j]))
            points.sort()
            X = list(range(len(vals[i])))
            plt.bar(X, [+a for a, b in points], facecolor=colors[i],
                    edgecolor=colors[i], label=labels[i])
            plt.bar(X, [-b for a, b in points], facecolor=colors[j],
                    edgecolor=colors[j], label=labels[j])
            plt.legend(loc='upper left')

    plt.savefig("figure.pdf")
    plt.savefig("figure.jpg")
    plt.close()


if __name__ == '__main__':
    main()
