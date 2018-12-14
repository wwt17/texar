from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import importlib
import os
import collections
import numpy as np
import tensorflow as tf
import texar as tx
import pickle
from utils import *
from munkres import *

flags = tf.flags
flags.DEFINE_string("config_data", "config_data_nba", "The data config.")
FLAGS = flags.FLAGS

config_data = importlib.import_module(FLAGS.config_data)

sent_fields = ['sent']
sd_fields = ['entry', 'attribute', 'value']
all_fields = sent_fields + sd_fields
ref_strs = ['', '_ref']

DataItem = collections.namedtuple('DataItem', sd_fields)


inf = int(1e9)
def calc_cost(a, b):
    if a.attribute != b.attribute:
        return inf
    if a.entry.isdigit():
        if b.entry.isdigit():
            return abs(int(a.entry) - int(b.entry))
        else:
            return inf
    else:
        if b.entry.isdigit():
            return inf
        else:
            return 0 if a.entry == b.entry else 1


def get_match(batch, verbose=False):
    batch_ = {}
    xs = []
    for ref_str in ref_strs:
        for field in all_fields:
            name = '{}{}'.format(field, ref_str)
            text_name = '{}_text'.format(name)
            batch_[text_name] = tx.utils.strip_special_tokens(
                batch[text_name], is_token_list=True)
            length_name = '{}_length'.format(name)
            batch_[length_name] = batch[length_name] - 2
        x = list(map(lambda _: DataItem(*_),
                     zip(*[batch_['{}{}_text'.format(field, ref_str)]
                           for field in sd_fields])))
        xs.append(x)

    if verbose:
        for name, value in batch_.items():
            print('{}: {}'.format(name, value))
        for x in xs:
            print(x)

    cost = [[calc_cost(x_i, x_j) for x_j in xs[1]] for x_i in xs[0]]
    match = Munkres().compute(cost)

    if verbose:
        print('{} matches:'.format(len(match)))
        for i, j in match:
            print('{}\t{}\t->{}'.format(xs[0][i], xs[1][j], cost[i][j]))

    return match


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()


    def _get_match(sess, mode):
        print('in _get_alignment')

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode),
        }

        with open('match.pkl', 'wb') as out_file:
            while True:
                try:
                    batch = sess.run(
                        data_batch,
                        feed_dict)
                    items = tuple(batch.items())
                    keys, values = zip(*items)
                    for values_ in zip(*values):
                        batch_ = {
                            key: value for key, value in zip(keys, values_)}
                        match = get_match(batch_)
                        pickle.dump(match, out_file)

                except tf.errors.OutOfRangeError:
                    break

        print('end _get_alignment')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        _get_match(sess, 'train')


if __name__ == '__main__':
    main()
