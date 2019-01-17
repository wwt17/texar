"""
Utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import texar as tx
from tensorflow.contrib.seq2seq import tile_batch

get_scope_name_of_train_op = 'train_{}'.format
get_scope_name_of_summary_op = 'summary_{}'.format


sent_fields = ['sent']
sd_fields = ['entry', 'attribute', 'value']
all_fields = sent_fields + sd_fields
ref_strs = ['', '_ref']

DataItem = collections.namedtuple('DataItem', sd_fields)

def list_strip_eos(list_, eos_token):
    """Strips EOS token from a list of lists of tokens.
    """
    list_strip = []
    for elem in list_:
        if eos_token in elem:
            elem = elem[:elem.index(eos_token)]
        list_strip.append(elem)
    return list_strip

def pack_sd(paired_texts):
    return [DataItem(*_) for _ in zip(*paired_texts)]

def batchize(func):
    def batchized_func(*inputs):
        return [func(*paired_inputs) for paired_inputs in zip(*inputs)]
    return batchized_func

def strip_special_tokens_of_list(text):
    return tx.utils.strip_special_tokens(text, is_token_list=True)

batch_strip_special_tokens_of_list = batchize(strip_special_tokens_of_list)

def corpus_bleu(list_of_references, hypotheses, **kwargs):
    return tx.evals.corpus_bleu_moses(
        list_of_references, hypotheses,
        lowercase=True, return_all=False,
        **kwargs)
