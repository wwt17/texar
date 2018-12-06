"""
Text Content Manipulation
3-gated copy net.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import importlib
import os
import tensorflow as tf
import texar as tx
from copy_net import CopyNetWrapper
from texar.core import get_train_op
from utils import *

flags = tf.flags
flags.DEFINE_string("config_data", "config_data_nba", "The data config.")
flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_train", "config_train", "The training config.")
flags.DEFINE_string("expr_name", "nba", "The experiment name. "
                    "Used as the directory name of run.")
flags.DEFINE_string("restore_from", "", "The specific checkpoint path to "
                    "restore from. If not specified, the latest checkpoint in "
                    "expr_name is used.")
FLAGS = flags.FLAGS

config_data = importlib.import_module(FLAGS.config_data)
config_model = importlib.import_module(FLAGS.config_model)
config_train = importlib.import_module(FLAGS.config_train)
expr_name = FLAGS.expr_name
restore_from = FLAGS.restore_from

dir_summary = os.path.join(expr_name, 'log')
dir_model = os.path.join(expr_name, 'ckpt')
dir_best = os.path.join(expr_name, 'ckpt-best')
ckpt_model = os.path.join(dir_model, 'model.ckpt')
ckpt_best = os.path.join(dir_best, 'model.ckpt')


def build_model(data_batch, data):
    batch_size, num_steps = [
        tf.shape(data_batch["value_text_ids"])[d] for d in range(2)]

    # embedders
    sent_vocab = data.vocab('sent')
    sent_embedder = tx.modules.WordEmbedder(
        vocab_size=sent_vocab.size, hparams=config_model.sent_embedder)
    entry_vocab = data.vocab('entry')
    entry_embedder = tx.modules.WordEmbedder(
        vocab_size=entry_vocab.size, hparams=config_model.sd_embedder)
    attribute_vocab = data.vocab('attribute')
    attribute_embedder = tx.modules.WordEmbedder(
        vocab_size=attribute_vocab.size, hparams=config_model.sd_embedder)
    value_vocab = data.vocab('entry')
    value_embedder = tx.modules.WordEmbedder(
        vocab_size=value_vocab.size, hparams=config_model.sd_embedder)

    # encoders
    sent_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.sent_encoder)
    structured_data_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.sd_encoder)

    # X & Y
    sents = data_batch['sent_text_ids'][:, :-1]
    entries = data_batch['entry_text_ids'][:, 1:-1]
    attributes = data_batch['attribute_text_ids'][:, 1:-1]
    values = data_batch['value_text_ids'][:, 1:-1]
    sent_embeds = sent_embedder(sents)  # [batch_size, num_steps, hidden_size]
    structured_data_embeds = tf.concat(
        [entry_embedder(entries),
         attribute_embedder(attributes),
         value_embedder(values)],
        axis=-1)  # [batch_size, tup_nums, hidden_size]
    sent_enc_outputs, _ = sent_encoder(sent_embeds)
    structured_data_enc_outputs, _ = structured_data_encoder(
        structured_data_embeds)

    # X' & Y'
    tplt_sents = data_batch['sent_ref_text_ids'][:, :-1]
    tplt_entries = data_batch['entry_ref_text_ids'][:, 1:-1]
    tplt_attributes = data_batch['attribute_ref_text_ids'][:, 1:-1]
    tplt_values = data_batch['value_ref_text_ids'][:, 1:-1]
    tplt_sent_embeds = sent_embedder(tplt_sents)  # [batch_size, num_steps, hidden_size]
    tplt_structured_data_embeds = tf.concat(
        [entry_embedder(tplt_entries),
         attribute_embedder(tplt_attributes),
         value_embedder(tplt_values)],
        axis=-1)
    tplt_sent_enc_outputs, _ = sent_encoder(tplt_sent_embeds)
    tplt_structured_data_enc_outputs, _ = structured_data_encoder(
        tplt_structured_data_embeds)

    # copy net decoder
    cell = tx.core.layers.get_rnn_cell(config_model.rnn_cell)
    copy_net_cell = CopyNetWrapper(
        cell=cell,
        template_encoder_states=tf.concat(tplt_sent_enc_outputs, -1),
        template_encoder_input_ids=sents,
        structured_data_encoder_states=tf.concat(structured_data_enc_outputs, -1),
        structured_data_encoder_input_ids=tf.concat([entries, attributes, values], axis=1),
        vocab_size=sent_vocab.size)
    decoder = tx.modules.BasicRNNDecoder(
        cell=copy_net_cell,
        output_layer=tf.identity,
        hparams=config_model.decoder)

    # teacher-forcing training
    tf_outputs, _, _ = decoder(
        decoding_strategy='train_greedy',
        inputs=sent_embeds,
        sequence_length=data_batch['sent_length'] - 1)

    # beam-search inference
    start_tokens = tf.ones_like(data_batch['sent_length']) * \
        data.vocab('sent').bos_token_id
    end_token = data.vocab('sent').eos_token_id

    bs_outputs, _, _ = tx.modules.beam_search_decode(
        decoder_or_cell=decoder,
        embedding=sent_embedder,
        start_tokens=start_tokens,
        end_token=end_token,
        beam_width=config_train.infer_beam_width,
        max_decoding_length=config_train.infer_max_decoding_length)

    # losses
    losses = {}

    # MLE loss
    losses['MLE'] = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch['sent_text_ids'][:, 1:],
        logits=tf_outputs.logits,
        sequence_length=data_batch['sent_length'] - 1)

    train_ops = {
        name: get_train_op(loss, hparams=config_train.train[name])
        for name, loss in losses.items()}

    return train_ops, bs_outputs


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()

    global_step = tf.train.get_or_create_global_step()

    train_ops, bs_outputs = build_model(data_batch, datasets['train'])

    summary_ops = {
        name: tf.summary.merge(
            tf.get_collection(
                tf.GraphKeys.SUMMARIES,
                scope=get_scope_name_of_train_op(name)),
            name=get_scope_name_of_summary_op(name))
        for name in train_ops.keys()}

    saver = tf.train.Saver(max_to_keep=None)

    best_ever_val_bleu = 0.


    def _save_to(directory, step):
        print('saving to {} ...'.format(directory))

        saved_path = saver.save(sess, directory, global_step=step)

        print('saved to {}'.format(saved_path))


    def _restore_from_path(ckpt_path):
        print('restoring from {} ...'.format(ckpt_path))

        saver.restore(sess, ckpt_path)

        print('done.')


    def _restore_from(directory):
        if os.path.exists(directory):
            ckpt_path = tf.train.latest_checkpoint(directory)

        else:
            print('cannot find checkpoint directory {}'.format(directory))


    def _train_epoch(sess, summary_writer, mode, train_op, summary_op):
        print('in _train_epoch')

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            data_iterator.handle: data_iterator.get_handle(sess, mode),
        }

        while True:
            try:
                loss, summary = sess.run((train_op, summary_op), feed_dict)

                step = tf.train.global_step(sess, global_step)

                summary_writer.add_summary(summary, step)

                if step % config_train.steps_per_eval == 0:
                    _eval_epoch(sess, summary_writer, 'val')

            except tf.errors.OutOfRangeError:
                break

        print('end _train_epoch')


    def _eval_epoch(sess, summary_writer, mode):
        print('in _eval_epoch with mode {}'.format(mode))

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode)
        }

        ref_hypo_pairs = []
        fetches = [
            data_batch['sent_text_ids'],
            bs_outputs.predicted_ids,
        ]

        while True:
            try:
                target_texts, output_ids = sess.run(fetches, feed_dict)
                target_texts = target_texts[:, 1:]
                output_ids = output_ids[:, :, 0]
                target_texts = tx.utils.strip_special_tokens(
                    target_texts.tolist(), is_token_list=True)
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids.tolist(), vocab=val_data.target_vocab,
                    join=False)

                ref_hypo_pairs.extend(zip(target_texts, output_texts))

            except tf.errors.OutOfRangeError:
                break

        refs, hypos = zip(*ref_hypo_pairs)
        refs = list(map(lambda x: [x], refs))
        bleu = corpus_bleu(refs, hypos)
        print('{} BLEU: {}'.format(mode, bleu))

        step = tf.train.global_step(sess, global_step)

        summary = tf.Summary()
        summary.value.add(tag='{}/BLEU'.format(mode), simple_value=bleu)
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

        if mode == 'val':
            if bleu > best_ever_val_bleu:
                best_ever_val_bleu = bleu
                print('updated best val bleu: {}'.format(bleu))

                _save_to(ckpt_best, step)

        print('end _eval_epoch')
        return bleu


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        if restore_from:
            _restore_from_path(restore_from)
        else:
            _restore_from(dir_model)

        summary_writer = tf.summary.FileWriter(
            dir_summary, sess.graph, flush_secs=30)

        epoch = 0
        while epoch < config_train.max_epochs:
            name = 'MLE'
            train_op = train_ops[name]
            summary_op = summary_ops[name]

            val_bleu = _eval_epoch(sess, summary_writer, 'val')
            test_bleu = _eval_epoch(sess, summary_writer, 'test')

            step = tf.train.global_step(sess, global_step)

            print('epoch: {} ({}), step: {}, val BLEU: {}, test BLEU: {}'\
                .format(epoch, name, step, val_bleu, test_bleu))

            _train_epoch(sess, summary_writer, 'train', train_op, summary_op)

            epoch += 1

            step = tf.train.global_step(sess, global_step)
            _save_to(ckpt_model, step)

        test_bleu = _eval_epoch(sess, summary_writer, 'test')
        print('epoch: {}, test BLEU: {}'.format(epoch, test_bleu))


if __name__ == '__main__':
    main()
