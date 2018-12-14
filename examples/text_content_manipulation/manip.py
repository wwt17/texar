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
import numpy as np
import tensorflow as tf
import texar as tx
import pickle
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
flags.DEFINE_boolean("align", False, "Whether it is to get alignment.")
flags.DEFINE_boolean("output_align", False, "Whether to output alignment.")
FLAGS = flags.FLAGS

if FLAGS.output_align:
    FLAGS.align = True
if FLAGS.align:
    FLAGS.attn = True
    FLAGS.copynet = False

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
    vocab = data.vocab('sent')


    id2str = '<{}>'.format
    bos_str, eos_str = map(id2str, (vocab.bos_token_id, vocab.eos_token_id))

    def single_bleu(ref, hypo):
        ref = [id2str(u if u != vocab.unk_token_id else -1) for u in ref]
        hypo = [id2str(u) for u in hypo]

        ref = tx.utils.strip_special_tokens(
            ' '.join(ref), strip_bos=bos_str, strip_eos=eos_str)
        hypo = tx.utils.strip_special_tokens(
            ' '.join(hypo), strip_eos=eos_str)

        return 0.01 * tx.evals.sentence_bleu(references=[ref], hypothesis=hypo)

    def batch_bleu(refs, hypos):
        return np.array(
            [single_bleu(ref, hypo) for ref, hypo in zip(refs, hypos)],
            dtype=np.float32)


    # losses
    losses = {}

    # embedders
    embedders = {
        name: tx.modules.WordEmbedder(
            vocab_size=data.vocab(name).size, hparams=hparams)
        for name, hparams in config_model.embedders.items()}

    # encoders
    sent_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.sent_encoder)
    sd_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.sd_encoder)


    def concat_encoder_outputs(outputs):
        return tf.concat(outputs, -1)


    sd_fields = ['entry', 'attribute', 'value']


    def encode(ref_str):
        sent_ids = data_batch['sent{}_text_ids'.format(ref_str)][:, :-1]
        sent_embeds = embedders['sent'](sent_ids)
        sent_sequence_length = data_batch['sent{}_length'.format(ref_str)] - 1
        sent_enc_outputs, _ = sent_encoder(
            sent_embeds, sequence_length=sent_sequence_length)
        sent_enc_outputs = concat_encoder_outputs(sent_enc_outputs)

        sd_ids = {
            field: data_batch['{}{}_text_ids'.format(field, ref_str)][:, 1:-1]
            for field in sd_fields}
        sd_embeds = tf.concat(
            [embedders[field](sd_ids[field]) for field in sd_fields],
            axis=-1)
        sd_sequence_length = data_batch[
            '{}{}_length'.format(sd_fields[0], ref_str)] - 2
        sd_enc_outputs, _ = sd_encoder(
            sd_embeds, sequence_length=sd_sequence_length)
        sd_enc_outputs = concat_encoder_outputs(sd_enc_outputs)

        return sent_ids, sent_embeds, sent_enc_outputs, sent_sequence_length, \
            sd_ids, sd_embeds, sd_enc_outputs, sd_sequence_length


    ref_strs = ['', '_ref']
    encode_results = [encode(ref_str) for ref_str in ref_strs]
    sent_ids, sent_embeds, sent_enc_outputs, sent_sequence_length, \
            sd_ids, sd_embeds, sd_enc_outputs, sd_sequence_length = \
        zip(*encode_results)

    # get rnn cell
    rnn_cell = tx.core.layers.get_rnn_cell(config_model.rnn_cell)


    def get_decoder(cell, tplt_ref_flag, sd_ref_flag, tgt_ref_flag,
                    beam_width=None):
        output_layer_params = \
            {'output_layer': tf.identity} if FLAGS.copynet else \
            {'vocab_size': vocab.size}

        if FLAGS.attn: # attention
            if tplt_ref_flag is not None and sd_ref_flag is not None:
                memory = tf.concat(
                    [sent_enc_outputs[tplt_ref_flag],
                     sd_enc_outputs[sd_ref_flag]],
                    axis=1)
                memory_sequence_length = None
            elif tplt_ref_flag is not None:
                memory = sent_enc_outputs[tplt_ref_flag]
                memory_sequence_length = sent_sequence_length[tplt_ref_flag]
            elif sd_ref_flag is not None:
                memory = sd_enc_outputs[sd_ref_flag]
                memory_sequence_length = sd_sequence_length[sd_ref_flag]
            else:
                raise Exception(
                    "Must specify either tplt_ref_flag or sd_ref_flag.")
            attention_decoder = tx.modules.AttentionRNNDecoder(
                cell=cell,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                hparams=config_model.attention_decoder,
                **output_layer_params)
            if not FLAGS.copynet:
                return attention_decoder
            cell = attention_decoder.cell if beam_width is None else \
                   attention_decoder._get_beam_search_cell(beam_width)

        if FLAGS.copynet: # copynet
            kwargs = {
                'tplt_encoder_input_ids': sent_ids[tplt_ref_flag],
                'tplt_encoder_states': sent_enc_outputs[tplt_ref_flag],
                'sd_encoder_input_ids': sd_ids[sd_ref_flag]['entry'],
                'sd_encoder_states': sd_enc_outputs[sd_ref_flag]}
            if tgt_ref_flag is not None:
                kwargs.update({
                'input_ids': data_batch[
                    'sent{}_text_ids'.format(ref_strs[tgt_ref_flag])][:, :-1]})
            if beam_width is not None:
                kwargs = {
                    name: tile_batch(value, beam_width)
                    for name, value in kwargs.items()}

            def get_get_copy_scores(memory_ids_states, output_size):
                memory_copy_states = [
                    tf.layers.dense(
                        memory_states,
                        units=output_size,
                        activation=tf.nn.tanh,
                        use_bias=False)
                    for _, memory_states in memory_ids_states]
                def get_copy_scores(outputs):
                    return [
                        tf.einsum("ijm,im->ij", memory_copy_state, outputs)
                        for memory_copy_state in memory_copy_states]
                return get_copy_scores

            cell = CopyNetWrapper(
                cell=cell, vocab_size=vocab.size,
                memory_ids_states=[
                    (kwargs['{}_encoder_input_ids'.format(t)],
                     kwargs['{}_encoder_states'.format(t)])
                    for t in ['tplt', 'sd']],
                input_ids=\
                    kwargs['input_ids'] if tgt_ref_flag is not None else None,
                get_get_copy_scores=get_get_copy_scores)

        decoder = tx.modules.BasicRNNDecoder(
            cell=cell, hparams=config_model.decoder,
            **output_layer_params)
        return decoder

    def get_decoder_and_outputs(
            cell, tplt_ref_flag, sd_ref_flag, tgt_ref_flag, params,
            beam_width=None):
        decoder = get_decoder(
            cell, tplt_ref_flag, sd_ref_flag, tgt_ref_flag,
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

    def teacher_forcing(cell, tplt_ref_flag, sd_ref_flag, loss_name):
        tgt_ref_flag = sd_ref_flag
        tgt_str = 'sent{}'.format(ref_strs[tgt_ref_flag])
        sequence_length = data_batch['{}_length'.format(tgt_str)] - 1
        decoder, tf_outputs, _, _ = get_decoder_and_outputs(
            cell, tplt_ref_flag, sd_ref_flag, tgt_ref_flag,
            {'decoding_strategy': 'train_greedy',
             'inputs': sent_embeds[tgt_ref_flag],
             'sequence_length': sequence_length})

        tgt_sent_ids = data_batch['{}_text_ids'.format(tgt_str)][:, 1:]
        loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=tgt_sent_ids,
            logits=tf_outputs.logits,
            sequence_length=sequence_length,
            average_across_batch=False)
        if config_train.add_bleu_weight and tplt_ref_flag is not None \
                and tgt_ref_flag is not None and tplt_ref_flag != tgt_ref_flag:
            w = tf.py_func(
                batch_bleu, [sent_ids[tplt_ref_flag], tgt_sent_ids],
                tf.float32, stateful=False, name='W_BLEU')
            w.set_shape(loss.get_shape())
            loss = w * loss
        loss = tf.reduce_mean(loss, 0)
        losses[loss_name] = loss

        return decoder, tf_outputs, loss


    def beam_searching(cell, tplt_ref_flag, sd_ref_flag, beam_width):
        start_tokens = tf.ones_like(data_batch['sent_length']) * \
            vocab.bos_token_id
        end_token = vocab.eos_token_id

        decoder, bs_outputs, _, _ = get_decoder_and_outputs(
            cell, tplt_ref_flag, sd_ref_flag, None,
            {'embedding': embedders['sent'],
             'start_tokens': start_tokens,
             'end_token': end_token,
             'max_decoding_length': config_train.infer_max_decoding_length},
            beam_width=config_train.infer_beam_width)

        return decoder, bs_outputs


    if FLAGS.align:
        decoder, tf_outputs, loss = teacher_forcing(
            rnn_cell, None, 1, 'ALIGN')
        losses['joint'] = loss

        tiled_decoder, bs_outputs = beam_searching(
            rnn_cell, None, 1, config_train.infer_beam_width)

    else:
        decoder, tf_outputs, loss = teacher_forcing(rnn_cell, 1, 0, 'MLE')
        rec_decoder, _, rec_loss = teacher_forcing(rnn_cell, 1, 1, 'REC')
        if config_train.rec_weight == 0:
            joint_loss = loss
        elif config_train.rec_weight == 1:
            joint_loss = rec_loss
        else:
            joint_loss = (1 - config_train.rec_weight) * loss \
                       + config_train.rec_weight * rec_loss
        losses['joint'] = joint_loss

        tiled_decoder, bs_outputs = beam_searching(
            rnn_cell, 1, 0, config_train.infer_beam_width)

    train_ops = {
        name: get_train_op(losses[name], hparams=config_train.train[name])
        for name in config_train.train}

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


    def _get_alignment(sess, mode):
        print('in _get_alignment')

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode),
        }

        with open('align.pkl', 'wb') as out_file:
            while True:
                try:
                    attention_scores = sess.run(
                        tf_outputs.attention_scores,
                        feed_dict)
                    pickle.dump(attention_scores, out_file)

                except tf.errors.OutOfRangeError:
                    break

        print('end _get_alignment')


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
            [data_batch['sent_text'], data_batch['sent_ref_text']],
            bs_outputs.predicted_ids,
        ]

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
                    ids=output_ids.tolist(), vocab=datasets[mode].vocab('sent'),
                    join=False)

                target_texts = list(zip(*target_texts))

                for ref, hypo in zip(target_texts, output_texts):
                    if cnt < 10:
                        print('cnt = {}'.format(cnt))
                        for i, s in enumerate(ref):
                            print('ref{}: {}'.format(i, ' '.join(s)))
                        print('hypo: {}'.format(' '.join(hypo)))
                    cnt += 1
                print('processed {} samples'.format(cnt))

                ref_hypo_pairs.extend(zip(target_texts, output_texts))

            except tf.errors.OutOfRangeError:
                break

        refs, hypos = zip(*ref_hypo_pairs)
        bleus = []
        get_bleu_name = '{}_BLEU'.format
        print('In {} mode:'.format(mode))
        for i in range(len(fetches[0])):
            refs_ = list(map(lambda ref: ref[i:i+1], refs))
            bleu = corpus_bleu(refs_, hypos)
            print('{}: {:.2f}'.format(get_bleu_name(i), bleu))
            bleus.append(bleu)

        step = tf.train.global_step(sess, global_step)

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

        if FLAGS.output_align:
            _get_alignment(sess, 'train')
            return

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
