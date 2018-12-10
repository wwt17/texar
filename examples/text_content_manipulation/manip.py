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
flags.DEFINE_boolean("copynet", False, "Whether to use copynet.")
flags.DEFINE_boolean("attn", False, "Whether to use attention.")
FLAGS = flags.FLAGS

assert not (FLAGS.copynet and FLAGS.attn)

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
        vocab_size=entry_vocab.size, hparams=config_model.entry_embedder)
    attribute_vocab = data.vocab('attribute')
    attribute_embedder = tx.modules.WordEmbedder(
        vocab_size=attribute_vocab.size, hparams=config_model.attribute_embedder)
    value_vocab = data.vocab('value')
    value_embedder = tx.modules.WordEmbedder(
        vocab_size=value_vocab.size, hparams=config_model.value_embedder)

    # encoders
    sent_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.sent_encoder)
    structured_data_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.sd_encoder)


    def concat_encoder_outputs(outputs):
        return tf.concat(outputs, -1)


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
    sent_enc_outputs = concat_encoder_outputs(
        sent_enc_outputs)
    structured_data_enc_outputs, _ = structured_data_encoder(
        structured_data_embeds)
    structured_data_enc_outputs = concat_encoder_outputs(
        structured_data_enc_outputs)

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
    tplt_sent_enc_outputs = concat_encoder_outputs(
        tplt_sent_enc_outputs)
    tplt_structured_data_enc_outputs, _ = structured_data_encoder(
        tplt_structured_data_embeds)
    tplt_structured_data_enc_outputs = concat_encoder_outputs(
        tplt_structured_data_enc_outputs)

    # get rnn cell
    rnn_cell = tx.core.layers.get_rnn_cell(config_model.rnn_cell)
    cell = rnn_cell
    decoder_params = {
        'cell': cell,
        'vocab_size': sent_vocab.size,
    }

    if FLAGS.copynet: # copynet
        cell = CopyNetWrapper(
            cell=cell,
            template_encoder_states=tplt_sent_enc_outputs,
            template_encoder_input_ids=tplt_sents,
            structured_data_encoder_states=structured_data_enc_outputs,
            structured_data_encoder_input_ids=entries,
            vocab_size=sent_vocab.size,
            input_ids=sents)
        decoder_params = {
            'cell': cell,
            'output_layer': tf.identity,
        }

    decoder_params['hparams'] = config_model.decoder

    if FLAGS.attn: # attention
        decoder_params['hparams'] = config_model.attention_decoder
        memory = tf.concat(
            [tplt_sent_enc_outputs, structured_data_enc_outputs],
            axis=1)
        decoder = tx.modules.AttentionRNNDecoder(
            memory=memory,
            memory_sequence_length=tf.ones(
                [tf.shape(memory)[0]], dtype=tf.int32) * tf.shape(memory)[1],
            **decoder_params)
    else:
        decoder = tx.modules.BasicRNNDecoder(
            **decoder_params)

    # teacher-forcing training
    tf_outputs, _, _ = decoder(
        decoding_strategy='train_greedy',
        inputs=sent_embeds,
        sequence_length=data_batch['sent_length'] - 1)

    # beam-search inference
    infer_beam_width = config_train.infer_beam_width
    tiled_cell = rnn_cell
    tiled_decoder = decoder

    if FLAGS.copynet:
        tiled_cell = CopyNetWrapper(
            cell=rnn_cell,
            template_encoder_states=tile_batch(
                tplt_sent_enc_outputs, infer_beam_width),
            template_encoder_input_ids=tile_batch(
                tplt_sents, infer_beam_width),
            structured_data_encoder_states=tile_batch(
                structured_data_enc_outputs, infer_beam_width),
            structured_data_encoder_input_ids=tile_batch(
                entries, infer_beam_width),
            vocab_size=sent_vocab.size,
            input_ids=sents)
        tiled_decoder = tx.modules.BasicRNNDecoder(
            cell=tiled_cell,
            output_layer=tf.identity,
            hparams=config_model.decoder)

    start_tokens = tf.ones_like(data_batch['sent_length']) * \
        data.vocab('sent').bos_token_id
    end_token = data.vocab('sent').eos_token_id

    bs_outputs, _, _ = tx.modules.beam_search_decode(
        decoder_or_cell=tiled_decoder,
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

    return train_ops, tf_outputs, bs_outputs


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()

    global_step = tf.train.get_or_create_global_step()

    train_ops, tf_outputs, bs_outputs = build_model(
        data_batch, datasets['train'])

    summary_ops = {
        name: tf.summary.merge(
            tf.get_collection(
                tf.GraphKeys.SUMMARIES,
                scope=get_scope_name_of_train_op(name)),
            name=get_scope_name_of_summary_op(name))
        for name in train_ops.keys()}

    saver = tf.train.Saver(max_to_keep=None)

    global best_ever_val_bleu
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
            _restore_from_path(ckpt_path)

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
        global best_ever_val_bleu

        print('in _eval_epoch with mode {}'.format(mode))

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode)
        }

        ref_hypo_pairs = []
        fetches = [
            data_batch['sent_text'],
            bs_outputs.predicted_ids,
        ]

        cnt = 0
        while True:
            try:
                target_texts, output_ids = sess.run(fetches, feed_dict)
                target_texts = target_texts[:, 1:]
                output_ids = output_ids[:, :, 0]
                target_texts = tx.utils.strip_special_tokens(
                    target_texts.tolist(), is_token_list=True)
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids.tolist(), vocab=datasets[mode].vocab('sent'),
                    join=False)

                for ref, hypo in zip(target_texts, output_texts):
                    if cnt < 10:
                        print('ref {}: {}'.format(cnt, ' '.join(ref)))
                        print('hyp {}: {}'.format(cnt, ' '.join(hypo)))
                    cnt += 1
                print('processed {} samples'.format(cnt))

                ref_hypo_pairs.extend(zip(target_texts, output_texts))

            except tf.errors.OutOfRangeError:
                break

        refs, hypos = zip(*ref_hypo_pairs)
        refs = list(map(lambda x: [x], refs))
        bleu = corpus_bleu(refs, hypos)
        print('{} BLEU: {:.2f}'.format(mode, bleu))

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

            print('epoch: {} ({}), step: {}, '
                  'val BLEU: {:.2f}, test BLEU: {:.2f}'.format(
                epoch, name, step, val_bleu, test_bleu))

            _train_epoch(sess, summary_writer, 'train', train_op, summary_op)

            epoch += 1

            step = tf.train.global_step(sess, global_step)
            _save_to(ckpt_model, step)

        test_bleu = _eval_epoch(sess, summary_writer, 'test')
        print('epoch: {}, test BLEU: {}'.format(epoch, test_bleu))


if __name__ == '__main__':
    main()
