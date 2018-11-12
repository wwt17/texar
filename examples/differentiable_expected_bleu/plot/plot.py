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

    for item in data:
        for i in range(len(item)):
            if i != 2:
                x = item[i]
                _x = x[0]
                assert all(map(lambda e: e == _x, x))
                item[i] = _x

    res = []
    for target_sent, bs_sent, sample_sents, debleu_loss, mle_loss in data:
        bs_bleu = sentence_bleu([target_sent], bs_sent)
        sample_bleu = mean([sentence_bleu([target_sent], sample_sent) for sample_sent in sample_sents])
        debleu, mle = map(lambda x: math.exp(-x), debleu_loss, mle_loss)
        _res = bs_bleu, sample_bleu, debleu, mle
        res.append(_res)
        #print("{}\n{}\n{}".format(' '.join(target_sent), ' '.join(bs_sent), ' '.join(sample_sents[0])))
        #print("{}\t{}\t{}".format(*_res))

    labels = ['beam search BLEU', 'sample mean BLEU', 'DEBLEU', 'MLE']
    colors = ['saddlebrown', 'mediumblue', 'darkgreen', 'crimson']

    vals = list(zip(*res))

    fig = plt.figure()
    fig.set_size_inches(10, 15)

    tot = (lambda x: x * (x-1) / 2)(len(labels))
    cnt = 0
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            cnt += 1
            ax = fig.add_subplot(tot, 2, cnt)
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])
            ax.scatter(vals[i], vals[j], s=1./3)
            cnt += 1
            ax = fig.add_subplot(tot, 2, cnt)
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
