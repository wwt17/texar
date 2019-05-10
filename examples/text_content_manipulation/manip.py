#!/usr/bin/env python3
"""
Text Content Manipulation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import importlib
import os
import numpy as np
import tensorflow as tf
import texar as tx
import pickle
from texar.modules import BasicCopyingMechanism, CopyNetWrapper
from texar.core import get_train_op
from utils import *

flags = tf.flags
flags.DEFINE_string("config_data", "config_data_nba", "The data config.")
flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_train", "config_train", "The training config.")
flags.DEFINE_float("rec_w", 0.8, "Weight of reconstruction loss.")
flags.DEFINE_float("rec_w_rate", 0., "Increasing rate of rec_w.")
flags.DEFINE_string("expr_name", "nba", "The experiment name. "
                    "Used as the directory name of run.")
flags.DEFINE_string("restore_from", "", "The specific checkpoint path to "
                    "restore from. If not specified, the latest checkpoint in "
                    "expr_name is used.")
flags.DEFINE_float("exact_cover_w", 0., "Weight of exact coverage loss.")
flags.DEFINE_float("eps", 1e-10, "epsilon used to avoid log(0).")
flags.DEFINE_integer("disabled_vocab_size", 1238, "Disabled vocab size.")
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


def get_optimistic_restore_variables(ckpt_path, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(ckpt_path)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([
        (var.name, var.name.split(':')[0]) for var in tf.global_variables()
        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        var = graph.get_tensor_by_name(var_name)
        var_shape = var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(var)
    return restore_vars


def get_optimistic_saver(ckpt_path, graph=tf.get_default_graph()):
    return tf.train.Saver(
        get_optimistic_restore_variables(ckpt_path, graph=graph))


def build_model(data_batch, data, step):
    batch_size, num_steps = [
        tf.shape(data_batch["x_value_text_ids"])[d] for d in range(2)]

    vocab = data.vocab('y_aux')

    # losses
    losses = {}

    # embedders
    embedders = {
        name: tx.modules.WordEmbedder(
            vocab_size=data.vocab(name).size, hparams=hparams)
        for name, hparams in config_model.embedders.items()}

    # encoders
    y_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.y_encoder)
    x_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.x_encoder)

    def concat_encoder_outputs(outputs):
        return tf.concat(outputs, -1)

    def encode(ref_flag):
        y_str = y_strs[ref_flag]
        y_ids = data_batch['{}_text_ids'.format(y_str)]
        y_embeds = embedders['y_aux'](y_ids)
        y_sequence_length = data_batch['{}_length'.format(y_str)]
        y_enc_outputs, _ = y_encoder(
            y_embeds, sequence_length=y_sequence_length)
        y_enc_outputs = concat_encoder_outputs(y_enc_outputs)

        x_str = x_strs[ref_flag]
        x_ids = {
            field: data_batch['{}_{}_text_ids'.format(x_str, field)][:, 1:-1]
            for field in x_fields}
        x_embeds = tf.concat(
            [embedders['x_{}'.format(field)](x_ids[field])
                for field in x_fields],
            axis=-1)
        x_sequence_length = data_batch[
            '{}_{}_length'.format(x_str, x_fields[0])] - 2
        x_enc_outputs, _ = x_encoder(
            x_embeds, sequence_length=x_sequence_length)
        x_enc_outputs = concat_encoder_outputs(x_enc_outputs)

        return y_ids, y_embeds, y_enc_outputs, y_sequence_length, \
            x_ids, x_embeds, x_enc_outputs, x_sequence_length

    encode_results = [encode(ref_flag) for ref_flag in range(2)]
    y_ids, y_embeds, y_enc_outputs, y_sequence_length, \
            x_ids, x_embeds, x_enc_outputs, x_sequence_length = \
        zip(*encode_results)

    # get rnn cell
    rnn_cell = tx.core.layers.get_rnn_cell(config_model.rnn_cell)

    def get_decoder(cell, y__flag, x_flag, tgt_flag, beam_width=None):
        # attention
        memory = tf.concat(
            [y_enc_outputs[y__flag],
             x_enc_outputs[x_flag]],
            axis=1)
        memory_sequence_length = None
        attention_decoder = tx.modules.AttentionRNNDecoder(
            cell=cell,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            output_layer=tf.identity,
            hparams=config_model.attention_decoder)
        cell = attention_decoder.cell if beam_width is None else \
               attention_decoder._get_beam_search_cell(beam_width)

        # copynet

        # y__copying_mechanism = BasicCopyingMechanism(
        #     num_units=cell.output_size,
        #     memory_id=y_ids[y__flag][:, 1:],
        #     memory_state=y_enc_outputs[y__flag][:, 1:],
        #     memory_sequence_length=y_sequence_length[y__flag] - 1,
        # )
        x_coverage_cell = tx.core.get_rnn_cell(
            config_model.coverage_rnn_cell)
        x_copying_mechanism = BasicCopyingMechanism(
            num_units=cell.output_size,
            memory_id=x_ids[x_flag]['value'],
            memory_state=x_enc_outputs[x_flag],
            memory_sequence_length=x_sequence_length[x_flag],
            activation=tf.tanh,
            coverage_state_dim=config_model.coverage_state_dim,
            coverage_cell=x_coverage_cell,
        )
        if beam_width is not None:
            x_copying_mechanism = \
                x_copying_mechanism._get_beam_search_copying_mechanism(
                    beam_width)

        vocab_size = vocab.size

        # disables generating data tokens
        def generating_layer(output):
            disabled_vocab_size = FLAGS.disabled_vocab_size
            score = tf.layers.dense(
                output, vocab_size - disabled_vocab_size, use_bias=False)
            disabled_score = (-np.inf) * tf.ones(
                tf.concat(
                    [tf.shape(score)[:-1], [FLAGS.disabled_vocab_size]], -1),
                dtype=score.dtype, name='disabled_score')
            score = tf.concat(
                [score[:, :4], disabled_score, score[:, 4:]], -1)
            return score

        cell = CopyNetWrapper(
            cell,
            x_copying_mechanism,
            vocab_size,
            generating_layer=generating_layer,
            output_layer=lambda probs: tf.log(probs + FLAGS.eps),
            copying_probability_history=
                config_model.copying_probability_history,
            coverage=config_model.coverage,
        )

        decoder = tx.modules.BasicRNNDecoder(
            cell=cell,
            output_layer=tf.identity,
            hparams=config_model.decoder
        )

        return decoder

    def get_decoder_and_outputs(
            cell, y__flag, x_flag, tgt_flag, params,
            beam_width=None):
        decoder = get_decoder(
            cell, y__flag, x_flag, tgt_flag,
            beam_width=beam_width)
        if beam_width is None:
            ret = decoder(**params)
        else:
            ret = tx.modules.beam_search_decode(
                decoder_or_cell=decoder,
                beam_width=beam_width,
                **params)
        return (decoder,) + ret

    get_decoder_and_outputs = tf.make_template(
        'get_decoder_and_outputs', get_decoder_and_outputs)

    def teacher_forcing(cell, y__flag, x_flag, loss_name):
        tgt_flag = x_flag
        tgt_str = y_strs[tgt_flag]
        sequence_length = data_batch['{}_length'.format(tgt_str)] - 1
        decoder, tf_outputs, final_state, _ = get_decoder_and_outputs(
            cell, y__flag, x_flag, tgt_flag,
            {'decoding_strategy': 'train_greedy',
             'inputs': y_embeds[tgt_flag],
             'sequence_length': sequence_length})

        tgt_sent_ids = data_batch['{}_text_ids'.format(tgt_str)][:, 1:]
        loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=tgt_sent_ids,
            logits=tf_outputs.logits,
            sequence_length=sequence_length,
            average_across_batch=False)
        loss = tf.reduce_mean(loss, 0)

        if FLAGS.exact_cover_w != 0:
            acc_copying_prob = tf.reduce_sum(
                final_state.copying_probability_history.stack(), axis=0)
            exact_coverage_loss = tf.reduce_mean(tf.reduce_sum(
                tx.utils.mask_sequences(
                    tf.square(acc_copying_prob - 1.),
                    x_sequence_length[x_flag]),
                1))
            print_xe_loss_op = tf.print(loss_name, 'xe loss:', loss)
            with tf.control_dependencies([print_xe_loss_op]):
                print_op = tf.print(
                    loss_name, 'exact coverage loss:', exact_coverage_loss)
                with tf.control_dependencies([print_op]):
                    loss += FLAGS.exact_cover_w * exact_coverage_loss

        losses[loss_name] = loss

        return decoder, tf_outputs, loss

    def beam_searching(cell, y__flag, x_flag, beam_width):
        start_tokens = tf.ones_like(data_batch['y_aux_length']) * \
            vocab.bos_token_id
        end_token = vocab.eos_token_id

        decoder, bs_outputs, _, _ = get_decoder_and_outputs(
            cell, y__flag, x_flag, None,
            {'embedding': embedders['y_aux'],
             'start_tokens': start_tokens,
             'end_token': end_token,
             'max_decoding_length': config_train.infer_max_decoding_length},
            beam_width=config_train.infer_beam_width)

        return decoder, bs_outputs

    decoder, tf_outputs, loss = teacher_forcing(rnn_cell, 1, 0, 'MLE')
    rec_decoder, _, rec_loss = teacher_forcing(rnn_cell, 1, 1, 'REC')
    rec_weight = FLAGS.rec_w + FLAGS.rec_w_rate * tf.cast(step, tf.float32)
    joint_loss = (1 - rec_weight) * loss + rec_weight * rec_loss
    losses['joint'] = joint_loss

    tiled_decoder, bs_outputs = beam_searching(
        rnn_cell, 1, 0, config_train.infer_beam_width)

    train_ops = {
        name: get_train_op(losses[name], hparams=config_train.train[name])
        for name in config_train.train}

    return train_ops, bs_outputs


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()

    global_step = tf.train.get_or_create_global_step()

    train_ops, bs_outputs, \
        = build_model(data_batch, datasets['train'], global_step)

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

        try:
            saver.restore(sess, ckpt_path)
        except tf.errors.NotFoundError:
            print('Some variables are missing. Try optimistically restoring.')
            (get_optimistic_saver(ckpt_path)).restore(sess, ckpt_path)

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

                print('step {:d}: loss = {:.6f}'.format(step, loss))

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

        step = tf.train.global_step(sess, global_step)

        ref_hypo_pairs = []
        fetches = [
            [data_batch['y_aux_text'], data_batch['y_ref_text']],
            bs_outputs.predicted_ids,
        ]

        if not os.path.exists(dir_model):
            os.makedirs(dir_model)

        hypo_file_name = os.path.join(
            dir_model, "hypos.step{}.{}.txt".format(step, mode))
        hypo_file = open(hypo_file_name, "w")

        cnt = 0
        while True:
            try:
                target_texts, output_ids = sess.run(fetches, feed_dict)
                target_texts = [
                    tx.utils.strip_special_tokens(
                        texts[:, 1:].tolist(), is_token_list=True)
                    for texts in target_texts]
                output_ids = output_ids[:, :, 0]
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids.tolist(), vocab=datasets[mode].vocab('y_aux'),
                    join=False)

                target_texts = list(zip(*target_texts))

                for ref, hypo in zip(target_texts, output_texts):
                    if cnt < 10:
                        print('cnt = {}'.format(cnt))
                        for i, s in enumerate(ref):
                            print('ref{}: {}'.format(i, ' '.join(s)))
                        print('hypo: {}'.format(' '.join(hypo)))
                    print(' '.join(hypo), file=hypo_file)
                    cnt += 1
                print('processed {} samples'.format(cnt))

                ref_hypo_pairs.extend(zip(target_texts, output_texts))

            except tf.errors.OutOfRangeError:
                break

        hypo_file.close()

        refs, hypos = zip(*ref_hypo_pairs)
        bleus = []
        get_bleu_name = '{}_BLEU'.format
        print('In {} mode:'.format(mode))
        for i in range(len(fetches[0])):
            refs_ = list(map(lambda ref: ref[i:i+1], refs))
            bleu = corpus_bleu(refs_, hypos)
            print('{}: {:.2f}'.format(get_bleu_name(i), bleu))
            bleus.append(bleu)

        summary = tf.Summary()
        for i, bleu in enumerate(bleus):
            summary.value.add(
                tag='{}/{}'.format(mode, get_bleu_name(i)), simple_value=bleu)
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

        bleu = bleus[0]
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
            name = 'joint'
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
