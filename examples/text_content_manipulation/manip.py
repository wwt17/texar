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
from get_xx import get_match
from get_xy import get_align
from ie import get_precrec
import rnn_decoders

flags = tf.flags
flags.DEFINE_string("config_data", "config_data_nba", "The data config.")
flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_train", "config_train", "The training config.")
flags.DEFINE_string("expr_name", "nba", "The experiment name. "
                    "Used as the directory name of run.")
flags.DEFINE_string("restore_from", "", "The specific checkpoint path to "
                    "restore from. If not specified, the latest checkpoint in "
                    "expr_name is used.")
flags.DEFINE_boolean("copy_x", False, "Whether to copy from x.")
flags.DEFINE_boolean("copy_y_", False, "Whether to copy from y'.")
flags.DEFINE_boolean("attn_x", False, "Whether to attend x.")
flags.DEFINE_boolean("attn_y_", False, "Whether to attend y'.")
flags.DEFINE_boolean("sd_path", False, "Whether to add structured data path.")
flags.DEFINE_float("sd_path_multiplicator", 1., "Structured data path multiplicator.")
flags.DEFINE_float("sd_path_addend", 0., "Structured data path addend.")
flags.DEFINE_boolean("align", False, "Whether it is to get alignment.")
flags.DEFINE_boolean("output_align", False, "Whether to output alignment.")
flags.DEFINE_boolean("verbose", False, "verbose.")
flags.DEFINE_boolean("eval_ie", False, "Whether evaluate IE.")
flags.DEFINE_integer("eval_ie_gpuid", 0, "ID of GPU on which IE runs.")
FLAGS = flags.FLAGS

copy_flag = FLAGS.copy_x or FLAGS.copy_y_
attn_flag = FLAGS.attn_x or FLAGS.attn_y_

if FLAGS.output_align:
    FLAGS.align = True

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


def print_alignment(data, sent, score):
    print(' ' * 20 + ' '.join(map('{:>12}'.format, data[0])))
    for j, sent_token in enumerate(sent[0]):
        print('{:>20}'.format(sent_token) + ' '.join(map(
            lambda x: '{:12.2e}'.format(x) if x != 0 else ' ' * 12,
            score[:, j])))


def batch_print_alignment(datas, sents, scores):
    datas, sents = map(
        lambda texts_lengths: map(
            lambda text_length:
                (text_length[0][:text_length[1]], text_length[1]),
            zip(*texts_lengths)),
        (datas, sents))
    for data, sent, score in zip(datas, sents, scores):
        score = score[:data[1], :sent[1]]
        print_alignment(data, sent, score)


def get_match_align(text00, text01, text02, text10, text11, text12, sent_text):
    """Combining match and align. All texts must not contain BOS.
    """
    matches = get_match(text00, text01, text02, text10, text11, text12)
    aligns = get_align(text10, text11, text12, sent_text)
    match = {i: j for i, j in matches}
    n = len(text00)
    m = len(sent_text)
    ret = np.zeros([n, m], dtype=np.float32)
    for i in range(n):
        try:
            k = match[i]
        except KeyError:
            continue
        align = aligns[k]
        ret[i][:len(align)] = align

    if FLAGS.verbose:
        print(' ' * 20 + ' '.join(map(
            '{:>12}'.format, strip_special_tokens_of_list(text00))))
        for j, sent_token in enumerate(strip_special_tokens_of_list(sent_text)):
            print('{:>20}'.format(sent_token) + ' '.join(map(
                lambda x: '{:>12}'.format(x) if x != 0 else ' ' * 12,
                ret[:, j])))

    return ret

def batch_get_match_align(*texts):
    return np.array(batchize(get_match_align)(*texts), dtype=np.float32)


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


    def encode(ref_str):
        sent_ids = data_batch['sent{}_text_ids'.format(ref_str)]
        sent_embeds = embedders['sent'](sent_ids)
        sent_sequence_length = data_batch['sent{}_length'.format(ref_str)]
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


    encode_results = [encode(ref_str) for ref_str in ref_strs]
    sent_ids, sent_embeds, sent_enc_outputs, sent_sequence_length, \
            sd_ids, sd_embeds, sd_enc_outputs, sd_sequence_length = \
        zip(*encode_results)

    # get rnn cell
    rnn_cell = tx.core.layers.get_rnn_cell(config_model.rnn_cell)


    def get_decoder(cell, y__ref_flag, x_ref_flag, tgt_ref_flag,
                    beam_width=None):
        output_layer_params = \
            {'output_layer': tf.identity} if copy_flag else \
            {'vocab_size': vocab.size}

        if attn_flag: # attention
            if FLAGS.attn_x and FLAGS.attn_y_:
                memory = tf.concat(
                    [sent_enc_outputs[y__ref_flag],
                     sd_enc_outputs[x_ref_flag]],
                    axis=1)
                memory_sequence_length = None
            elif FLAGS.attn_y_:
                memory = sent_enc_outputs[y__ref_flag]
                memory_sequence_length = sent_sequence_length[y__ref_flag]
            elif FLAGS.attn_x:
                memory = sd_enc_outputs[x_ref_flag]
                memory_sequence_length = sd_sequence_length[x_ref_flag]
            else:
                raise Exception(
                    "Must specify either y__ref_flag or x_ref_flag.")
            attention_decoder = rnn_decoders.AttentionRNNDecoder(
                cell=cell,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                hparams=config_model.attention_decoder,
                **output_layer_params)
            if not copy_flag:
                return attention_decoder
            cell = attention_decoder.cell if beam_width is None else \
                   attention_decoder._get_beam_search_cell(beam_width)

        if copy_flag: # copynet
            kwargs = {
                'y__ids': sent_ids[y__ref_flag][:, 1:],
                'y__states': sent_enc_outputs[y__ref_flag][:, 1:],
                'y__lengths': sent_sequence_length[y__ref_flag] - 1,
                'x_ids': sd_ids[x_ref_flag]['entry'],
                'x_states': sd_enc_outputs[x_ref_flag],
                'x_lengths': sd_sequence_length[x_ref_flag],
            }

            if tgt_ref_flag is not None:
                kwargs.update({
                'input_ids': data_batch[
                    'sent{}_text_ids'.format(ref_strs[tgt_ref_flag])][:, :-1]})

            memory_prefixes = []

            if FLAGS.copy_y_:
                memory_prefixes.append('y_')

            if FLAGS.copy_x:
                memory_prefixes.append('x')

            if FLAGS.sd_path:
                assert FLAGS.copy_x

                memory_prefixes.append('y_')

                texts = []
                for ref_flag in [x_ref_flag, y__ref_flag]:
                    texts.extend(data_batch['{}{}_text'.format(field, ref_strs[ref_flag])][:, 1:-1]
                                 for field in sd_fields)
                texts.extend(data_batch['{}{}_text'.format(field, ref_strs[y__ref_flag])][:, 1:]
                             for field in sent_fields)
                match_align = tf.py_func(
                    batch_get_match_align, texts, tf.float32, stateful=False,
                    name='match_align')
                match_align.set_shape(
                    [texts[-1].shape[0], texts[0].shape[1], texts[-1].shape[1]])

            if beam_width is not None:
                kwargs = {
                    name: tile_batch(value, beam_width)
                    for name, value in kwargs.items()}
                if FLAGS.sd_path:
                    match_align = tile_batch(match_align, beam_width)

            def get_get_copy_scores(memory_ids_states_lengths, output_size):
                memory_copy_states = [
                    tf.layers.dense(
                        memory_states,
                        units=output_size,
                        activation=tf.nn.tanh,
                        use_bias=False)
                    for _, memory_states, _ in memory_ids_states_lengths]

                def get_copy_scores(query):
                    ret = []

                    if FLAGS.copy_y_:
                        ret_y_ = tf.einsum("bim,bm->bi", memory_copy_states[len(ret)], query)
                        ret.append(ret_y_)

                    if FLAGS.copy_x:
                        ret_x = tf.einsum("bim,bm->bi", memory_copy_states[len(ret)], query)
                        ret.append(ret_x)

                    if FLAGS.sd_path:
                        ret_sd_path = FLAGS.sd_path_multiplicator * \
                            tf.einsum("bi,bij->bj", ret_x, match_align) \
                            + FLAGS.sd_path_addend
                        ret.append(ret_sd_path)

                    return ret

                return get_copy_scores

            cell = CopyNetWrapper(
                cell=cell, vocab_size=vocab.size,
                memory_ids_states_lengths=[
                    tuple(kwargs['{}_{}'.format(prefix, s)]
                          for s in ('ids', 'states', 'lengths'))
                    for prefix in memory_prefixes],
                input_ids=\
                    kwargs['input_ids'] if tgt_ref_flag is not None else None,
                get_get_copy_scores=get_get_copy_scores)

        decoder = rnn_decoders.BasicRNNDecoder(
            cell=cell, hparams=config_model.decoder,
            **output_layer_params)
        return decoder

    def get_decoder_and_outputs(
            cell, y__ref_flag, x_ref_flag, tgt_ref_flag, params,
            beam_width=None):
        decoder = get_decoder(
            cell, y__ref_flag, x_ref_flag, tgt_ref_flag,
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

    def teacher_forcing(cell, y__ref_flag, x_ref_flag, loss_name):
        tgt_ref_flag = x_ref_flag
        tgt_str = 'sent{}'.format(ref_strs[tgt_ref_flag])
        sequence_length = data_batch['{}_length'.format(tgt_str)] - 1
        decoder, tf_outputs, _, tf_lengths = get_decoder_and_outputs(
            cell, y__ref_flag, x_ref_flag, tgt_ref_flag,
            {'decoding_strategy': 'train_greedy',
             'inputs': sent_embeds[tgt_ref_flag],
             'sequence_length': sequence_length})

        tgt_sent_ids = data_batch['{}_text_ids'.format(tgt_str)][:, 1:]
        loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=tgt_sent_ids,
            logits=tf_outputs.logits,
            sequence_length=sequence_length,
            average_across_batch=False)
        if config_train.add_bleu_weight and y__ref_flag is not None \
                and tgt_ref_flag is not None and y__ref_flag != tgt_ref_flag:
            w = tf.py_func(
                batch_bleu, [sent_ids[y__ref_flag], tgt_sent_ids],
                tf.float32, stateful=False, name='W_BLEU')
            w.set_shape(loss.get_shape())
            loss = w * loss
        loss = tf.reduce_mean(loss, 0)
        losses[loss_name] = loss

        return decoder, tf_outputs, tf_lengths, loss

    start_tokens = tf.ones_like(data_batch['sent_length']) * \
        vocab.bos_token_id
    end_token = vocab.eos_token_id

    def infer_greedy(cell, y__ref_flag, x_ref_flag):
        tgt_ref_flag = x_ref_flag
        tgt_str = 'sent{}'.format(ref_strs[tgt_ref_flag])

        decoder, outputs, _, lengths = get_decoder_and_outputs(
            cell, y__ref_flag, x_ref_flag, None,
            {'decoding_strategy': 'infer_greedy',
             'embedding': embedders['sent'],
             'start_tokens': start_tokens,
             'end_token': end_token,
             'max_decoding_length': config_train.infer_max_decoding_length})

        tgt_sent_ids = data_batch['{}_text_ids'.format(tgt_str)][:, 1:]
        return decoder, outputs, lengths

    def beam_searching(cell, y__ref_flag, x_ref_flag, beam_width):
        decoder, bs_outputs, _, _ = get_decoder_and_outputs(
            cell, y__ref_flag, x_ref_flag, None,
            {'embedding': embedders['sent'],
             'start_tokens': start_tokens,
             'end_token': end_token,
             'max_decoding_length': config_train.infer_max_decoding_length},
            beam_width=config_train.infer_beam_width)

        return decoder, bs_outputs


    def build_align():
        ref_str = ref_strs[1]
        sent_str = 'sent{}'.format(ref_str)
        sent_texts = data_batch['{}_text'.format(sent_str)][:, 1:-1]
        sent_ids = data_batch['{}_text_ids'.format(sent_str)][:, 1:-1]
        #TODO: Here we simply use the embedder previously constructed,
        #therefore it's shared. We have to construct a new one here if we'd
        #like to get align on the fly.
        sent_embeds = embedders['sent'](sent_ids)
        sent_sequence_length = data_batch['{}_length'.format(sent_str)] - 2
        sent_enc_outputs, _ = sent_encoder(
            sent_embeds, sequence_length=sent_sequence_length)
        sent_enc_outputs = concat_encoder_outputs(sent_enc_outputs)

        sd_field = sd_fields[0]
        sd_str = '{}{}'.format(sd_field, ref_str)
        sd_texts = data_batch['{}_text'.format(sd_str)][:, :-1]
        sd_ids = data_batch['{}_text_ids'.format(sd_str)]
        tgt_sd_ids = sd_ids[:, 1:]
        sd_ids = sd_ids[:, :-1]
        sd_sequence_length = data_batch['{}_length'.format(sd_str)] - 1
        sd_embedder = embedders[sd_field]

        rnn_cell = tx.core.layers.get_rnn_cell(config_model.align_rnn_cell)
        attention_decoder = tx.modules.AttentionRNNDecoder(
            cell=rnn_cell,
            memory=sent_enc_outputs,
            memory_sequence_length=sent_sequence_length,
            vocab_size=vocab.size,
            hparams=config_model.align_attention_decoder)

        tf_outputs, _, tf_sequence_length = attention_decoder(
            decoding_strategy='train_greedy',
            inputs=sd_ids,
            embedding=sd_embedder,
            sequence_length=sd_sequence_length)

        loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=tgt_sd_ids,
            logits=tf_outputs.logits,
            sequence_length=sd_sequence_length)

        start_tokens = tf.ones_like(sd_sequence_length) * vocab.bos_token_id
        end_token = vocab.eos_token_id
        bs_outputs, _, _ = tx.modules.beam_search_decode(
            decoder_or_cell=attention_decoder,
            embedding=sd_embedder,
            start_tokens=start_tokens,
            end_token=end_token,
            max_decoding_length=config_train.infer_max_decoding_length,
            beam_width=config_train.infer_beam_width)

        return (sent_texts, sent_sequence_length), (sd_texts, sd_sequence_length),\
               loss, tf_outputs, bs_outputs


    decoder, tf_outputs, tf_lengths, loss = teacher_forcing(rnn_cell, 1, 0, 'MLE')
    rec_decoder, _, rec_lengths, rec_loss = teacher_forcing(rnn_cell, 1, 1, 'REC')
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

    align_sents, align_sds, align_loss, align_tf_outputs, align_bs_outputs = \
        build_align()
    losses['align'] = align_loss

    train_ops = {
        name: get_train_op(losses[name], hparams=config_train.train[name])
        for name in config_train.train}

    _, greedy_outputs, greedy_lengths = infer_greedy(rnn_cell, 1, 0)

    return train_ops, bs_outputs, tf_outputs, tf_lengths, greedy_outputs, greedy_lengths, \
           align_sents, align_sds, align_tf_outputs, align_bs_outputs


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()

    global_step = tf.train.get_or_create_global_step()

    train_ops, bs_outputs, tf_outputs, tf_lengths, greedy_outputs, greedy_lengths, \
            align_sents, align_sds, align_tf_outputs, align_bs_outputs \
        = build_model(data_batch, datasets['train'])

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


    def _get_alignment(sess, mode):
        print('in _get_alignment')

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            data_iterator.handle: data_iterator.get_handle(sess, mode),
        }

        fetches = [
            align_sds,
            align_sents,
            align_tf_outputs.attention_scores]

        with open('align.pkl', 'wb') as out_file:
            while True:
                try:
                    fetched = sess.run(fetches, feed_dict)
                    batch_print_alignment(*fetched)

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
        vocab = datasets['train'].vocab('sent')

        while True:
            try:
                loss, summary, lengths, copy_probs, Zs, path_weights, batch, gen_texts = sess.run((
                        train_op, summary_op, greedy_lengths,
                        greedy_outputs.cell_state.copy_probs,
                        greedy_outputs.cell_state.Zs,
                        greedy_outputs.cell_state.path_weights,
                        data_batch,
                        vocab.map_ids_to_tokens(greedy_outputs.sample_id),
                    ), feed_dict)
                entry_texts = batch['entry_text'][:, 1:]
                entry_ref_texts = batch['entry_ref_text'][:, 1:]
                sent_texts = batch['sent_text'][:, 1:]
                sent_ref_texts = batch['sent_ref_text'][:, 1:]
                all_name_texts = [
                    ("x", entry_texts),
                    ("x'", entry_ref_texts),
                    ("y", sent_texts),
                    ("y'", sent_ref_texts),
                    ("y^", gen_texts),
                ]
                text_names, all_texts = map(list, zip(*all_name_texts))
                cnt = len(copy_probs)
                for _ in zip(*([lengths] + copy_probs + Zs + [path_weights] + all_texts)):
                    steps, _, texts = _[0], _[1:-len(all_texts)], _[-len(all_texts):]
                    texts = dict(zip(text_names, texts))
                    texts["y^"] = texts["y^"][:steps]
                    for name in text_names:
                        print("{:<2}: {}".format(name, ' '.join(texts[name])))
                    copy_names = []
                    if FLAGS.copy_y_:
                        copy_names.append("y'")
                    if FLAGS.copy_x:
                        copy_names.append("x")
                    if FLAGS.sd_path:
                        copy_names.append("y'")
                    copy_texts = [texts[name] for name in copy_names]
                    print('decode steps: {}'.format(steps))
                    for step, __ in enumerate(zip(*_)):
                        if step >= steps:
                            break
                        __, path_weights__ = __[:-1], __[-1]
                        print('path_weights: {}'.format(' '.join(map('{:.2f}'.format, path_weights__))))
                        probs, zs = __[:cnt], __[cnt:]
                        print('zs: {}'.format(' '.join(map('{:.2f}'.format, zs))))
                        for name, prob, text in zip(copy_names, probs, copy_texts):
                            print('{name:<2}: {sum:.2f}\t{max:.2f}\t{argmax}'.format(
                                name=name,
                                sum=np.sum(prob),
                                max=np.max(prob),
                                argmax=text[np.argmax(prob)],
                            ))
                        print('result: {}'.format(texts["y^"][step]))

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

        step = tf.train.global_step(sess, global_step)

        ref_hypo_pairs = []
        fetches = [
            [data_batch['sent_text'], data_batch['sent_ref_text']],
            bs_outputs.predicted_ids,
        ] if not FLAGS.align else [
            [data_batch['entry_text'], data_batch['entry_ref_text']],
            align_bs_outputs.predicted_ids,
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
                    ids=output_ids.tolist(), vocab=datasets[mode].vocab('sent'),
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

        if FLAGS.eval_ie:
            gold_file_name = os.path.join(
                config_data.dst_dir, "gold.{}.txt".format(
                    config_data.mode_to_filemode[mode]))
            inter_file_name = "{}.h5".format(hypo_file_name[:-len(".txt")])
            prec, rec = get_precrec(
                gold_file_name, hypo_file_name, inter_file_name,
                gpuid=FLAGS.eval_ie_gpuid)

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
        if FLAGS.eval_ie:
            for name, value in {'precision': prec, 'recall': rec}.items():
                summary.value.add(tag='{}/{}'.format(mode, name),
                                  simple_value=value)
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
            name = 'align' if FLAGS.align else 'joint'
            train_op = train_ops[name]
            summary_op = summary_ops[name]

            val_bleu = 0. # _eval_epoch(sess, summary_writer, 'val')
            test_bleu = 0. # _eval_epoch(sess, summary_writer, 'test')

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
