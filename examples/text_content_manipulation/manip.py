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


def print_alignment(data0, data1, data2, sent, score):
    datas = [data0, data1, data2]
    for data in datas:
        print(' ' * 20 + ' '.join(map('{:>12}'.format, data)))
    def float2str(x):
        s = '{:12.2f}'.format(x)
        if s == '{:12.2f}'.format(0.):
            s = ' ' * 12
        return s
    for j, sent_token in enumerate(sent):
        print('{:>20}'.format(sent_token) + ' '.join(map(
            float2str, score[:, j])))


def strip_print_alignment(data0, data1, data2, sent, score):
    data0, data1, data2, sent = map(
        strip_special_tokens_of_list, (data0, data1, data2, sent))
    len_data = len(data0)
    print_alignment(data0, data1, data2, sent, score[:len_data])


batch_print_alignment = batchize(strip_print_alignment)


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
            attention_decoder = tx.modules.AttentionRNNDecoder(
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

        decoder = tx.modules.BasicRNNDecoder(
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
        decoder, tf_outputs, _, _ = get_decoder_and_outputs(
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

        return decoder, tf_outputs, loss


    def beam_searching(cell, y__ref_flag, x_ref_flag, beam_width):
        start_tokens = tf.ones_like(data_batch['sent_length']) * \
            vocab.bos_token_id
        end_token = vocab.eos_token_id

        decoder, bs_outputs, _, _ = get_decoder_and_outputs(
            cell, y__ref_flag, x_ref_flag, None,
            {'embedding': embedders['sent'],
             'start_tokens': start_tokens,
             'end_token': end_token,
             'max_decoding_length': config_train.infer_max_decoding_length},
            beam_width=config_train.infer_beam_width)

        return decoder, bs_outputs


    def build_align(align_model='predict'):
        # align_model = copredict: predict three fields together
        # align_model = predict the third field

        used_sent_fields = sent_fields
        sent_field = used_sent_fields[0]
        input_sd_fields = ['entry', 'value']
        output_sd_fields = ['attribute']

        used_sd_fields = input_sd_fields + output_sd_fields
        used_fields = sent_fields + sd_fields
        ref_str = ref_strs[1]
        sent_str = '{}{}'.format(sent_field, ref_str)

        # embedders
        embedders = {
            name: tx.modules.WordEmbedder(
                vocab_size=data.vocab(name).size,
                hparams=config_model.embedders[name])
            for name in used_fields}

        sent_texts = data_batch['{}_text'.format(sent_str)][:, 1:-1]
        sent_ids = data_batch['{}_text_ids'.format(sent_str)][:, 1:-1]
        sent_embeds = embedders[sent_field](sent_ids)
        sent_sequence_length = data_batch['{}_length'.format(sent_str)] - 2
        sent_enc_outputs, _ = sent_encoder(
            sent_embeds, sequence_length=sent_sequence_length)
        sent_enc_outputs = concat_encoder_outputs(sent_enc_outputs)

        sd_texts = {}
        input_sd_embeds = {}
        target_sd_ids = {}
        sd_sequence_lengths = {}

        for sd_field in input_sd_fields:
            sd_str = '{}{}'.format(sd_field, ref_str)
            sd_texts[sd_field] = data_batch['{}_text'.format(sd_str)]
            sd_ids = data_batch['{}_text_ids'.format(sd_str)]
            input_sd_embeds[sd_field] = embedders[sd_field](sd_ids[:, :-1])
            sd_sequence_lengths[sd_field] = data_batch['{}_length'.format(sd_str)] - 1

        for sd_field in output_sd_fields:
            sd_str = '{}{}'.format(sd_field, ref_str)
            sd_texts[sd_field] = data_batch['{}_text'.format(sd_str)]
            sd_ids = data_batch['{}_text_ids'.format(sd_str)]
            target_sd_ids[sd_field] = sd_ids[:, 1:]
            sd_sequence_lengths[sd_field] = data_batch['{}_length'.format(sd_str)] - 1

        input_sd_embeds = tf.concat(
            [input_sd_embeds[sd_field] for sd_field in input_sd_fields], -1)

        rnn_cell = tx.core.layers.get_rnn_cell(config_model.align_rnn_cell)


        attention_decoder = tx.modules.AttentionRNNDecoder(
            cell=rnn_cell,
            memory=sent_enc_outputs,
            memory_sequence_length=sent_sequence_length,
            output_layer=tf.identity,
            hparams=config_model.align_attention_decoder)

        tf_outputs, _, tf_sequence_length = attention_decoder(
            decoding_strategy='train_greedy',
            inputs=input_sd_embeds,
            embedding=None,
            sequence_length=sd_sequence_lengths[used_sd_fields[0]])
        cell_outputs = tf_outputs.cell_output

        projection_name = 'align_projection_{}'.format

        sd_field = output_sd_fields[0]
        projection = tf.layers.Dense(
            data.vocab(sd_field).size,
            name=projection_name(sd_field))
        """
        print_op = tf.print(
            [tf.shape(input_sd_embeds),
             sd_sequence_lengths[used_sd_fields[0]],
             tf.shape(sent_enc_outputs)])
        """
        logits = projection(cell_outputs)
        preds = tf.argmax(logits, axis=-1)
        labels = tf.cast(target_sd_ids[sd_field], preds.dtype)
        is_target = tf.to_float(tf.not_equal(labels, 
            data.vocab(sd_field).pad_token_id))

        matched_cnts = tf.reduce_sum(
            tf.to_float(tf.equal(preds, labels)) * is_target,
            axis=1)
        valid_cnts = tf.reduce_sum(is_target, axis=1)
        
        # with tf.control_dependencies([print_op]):
        loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=target_sd_ids[sd_field],
            logits=logits,
            sequence_length=sd_sequence_lengths[sd_field])

        return (sent_texts, sent_sequence_length), (sd_texts, sd_sequence_lengths),\
               loss, tf_outputs, preds, matched_cnts, valid_cnts


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

    align_sents, align_sds, align_loss, align_tf_outputs, align_preds, \
        align_matched, align_valids = \
        build_align()
    losses['align'] = align_loss

    train_ops = {
        name: get_train_op(losses[name], hparams=config_train.train[name])
        for name in config_train.train}

    return train_ops, bs_outputs, \
           align_sents, align_sds, align_tf_outputs, align_preds,\
           align_matched, align_valids


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()

    global_step = tf.train.get_or_create_global_step()

    train_ops, bs_outputs, \
            align_sents, align_sds, align_tf_outputs, align_preds,\
            align_matched, align_valids\
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

        with open('align.pkl', 'wb') as out_file:
            while True:
                try:
                    align_sds_, align_sents_, attention_scores \
                        = sess.run(
                        (align_sds,
                         align_sents,
                         align_tf_outputs.attention_scores),
                        feed_dict)
                    args = []
                    for field in sd_fields:
                        args.append(align_sds_[0][field])
                    args.append(align_sents_[0])
                    args.append(attention_scores)
                    batch_print_alignment(*args)

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
                loss, _m, _v, summary = sess.run(
                    (train_op, 
                     align_matched,
                     align_valids,
                     summary_op),
                    feed_dict)
                _accu = np.sum(_m) / np.sum(_v)
                step = tf.train.global_step(sess, global_step)

                _summary = tf.Summary()
                _summary.value.add(
                    tag='{}/align_accu'.format(mode),
                    simple_value=_accu)
                summary_writer.add_summary(_summary, step)
                summary_writer.add_summary(summary, step)

            except tf.errors.OutOfRangeError:
                break

        print('end _train_epoch')


    def _eval_epoch(sess, summary_writer, mode):
        print('in _eval_epoch with mode {}'.format(mode))

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode),
        }
        _matched_cnts = []
        _valid_cnts = []
        _labels = []
        _vocab = datasets['train'].vocab('attribute')
        _map_ids_to_tokens = _vocab.map_ids_to_tokens_py
        _predictions = []
        while True:
            try:
                _align_sds, _preds, _matched, _valids = sess.run(
                    (align_sds, align_preds, align_matched, align_valids),
                    feed_dict)
                _predictions.extend(
                    _map_ids_to_tokens(_preds).tolist()
                )
                _matched_cnts.extend(_matched.tolist())
                _valid_cnts.extend(_valids.tolist())
                _labels.extend(
                    _align_sds[0]['attribute'][:, 1:].tolist())
            except tf.errors.OutOfRangeError:
                break
        fname = os.path.join(expr_name, '{}_output'.format(mode))
        _predictions = list_strip_eos(_predictions,
                                      _vocab.eos_token)
        _labels = list_strip_eos(_labels,
                                 _vocab.eos_token)
                    
        _predictions = tx.utils.str_join(_predictions)
        _labels = tx.utils.str_join(_labels)
        hyp_fn, ref_fn = tx.utils.write_paired_text(
            _predictions, _labels, fname, mode='s')
        with open(os.path.join(expr_name, '{}_accu.txt'.format(mode)),'w+') as fout:
            for _m, _v in zip(_matched_cnts, _valid_cnts):
                fout.write('matched:{} valid:{} accu:{}\n'.format(
                _m,
                _v,
                _m/float(_v)))
        eval_accu = np.sum(_matched_cnts) / np.sum(_valid_cnts)
        print('{} accuracy:{}'.format(mode, eval_accu))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    with tf.Session(config=tf_config) as sess:
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

        sess.graph.finalize()

        summary_writer = tf.summary.FileWriter(
            dir_summary, sess.graph, flush_secs=30)

        epoch = 0
        while epoch < config_train.max_epochs:
            name = 'align' if FLAGS.align else 'joint'
            train_op = train_ops[name]
            summary_op = summary_ops[name]

            _eval_epoch(sess, summary_writer, 'val')
            _eval_epoch(sess, summary_writer, 'test')

            step = tf.train.global_step(sess, global_step)
            _train_epoch(sess, summary_writer, 'train', train_op, summary_op)

            epoch += 1

            step = tf.train.global_step(sess, global_step)
            _save_to(ckpt_model, step)

if __name__ == '__main__':
    main()
