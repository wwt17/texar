"""
Utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import texar as tx


get_scope_name_of_train_op = 'train_{}'.format
get_scope_name_of_summary_op = 'summary_{}'.format


def corpus_bleu(list_of_references, hypotheses, **kwargs):
    return tx.evals.corpus_bleu_moses(
        list_of_references, hypotheses,
        lowercase=True, return_all=False,
        **kwargs)
