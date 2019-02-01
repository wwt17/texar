# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DEBLEU.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#pylint: disable=invalid-name, too-many-arguments, too-many-locals

import importlib
import os
import pickle
import random
import torchtext.data
import tensorflow as tf
import texar as tx
from texar.utils import transformer_utils
import numpy as np

from utils import data_utils, utils
from utils.preprocess import pad_token_id, bos_token_id, eos_token_id, \
    unk_token_id


flags = tf.flags

flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_data", "config_data_iwslt14_de-en",
                    "The dataset config.")
flags.DEFINE_string("config_train", "config_train", "The training config.")
flags.DEFINE_string("expr_name", "iwslt14_de-en", "The experiment name. "
                    "Used as the directory name of run.")
flags.DEFINE_string("restore_from", "", "The specific checkpoint path to "
                    "restore from. If not specified, the latest checkpoint in "
                    "expr_name is restored.")
flags.DEFINE_boolean("reinitialize", True, "Whether to reinitialize the state "
                     "of the optimizers before training and after annealing.")
flags.DEFINE_string("pickle_prefix", "", "pickle dump val/test result prefix.")

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
config_data = importlib.import_module(FLAGS.config_data)
config_train = importlib.import_module(FLAGS.config_train)
expr_name = FLAGS.expr_name
restore_from = FLAGS.restore_from
reinitialize = FLAGS.reinitialize
pickle_prefix = FLAGS.pickle_prefix
need_pickle = bool(pickle_prefix)
phases = config_train.phases

xe_names = ('xe',)
debleu_names = ('debleu',)
pg_names = ('pg_grd', 'pg_msp')
all_names = xe_names + debleu_names + pg_names

dir_model = os.path.join(expr_name, 'ckpt')
dir_best = os.path.join(expr_name, 'ckpt-best')
ckpt_model = os.path.join(dir_model, 'model.ckpt')
ckpt_best = os.path.join(dir_best, 'model.ckpt')


def get_scope_by_name(tensor):
    return tensor.name[: tensor.name.rfind('/') + 1]


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


def strip_eos_id(sent_id):
    try:
        return sent_id[:sent_id.index(eos_token_id)]
    except ValueError:
        return sent_id


def corpus_bleu(refs, hypos):
    def join_and_decode(sent):
        sent = ' '.join(sent)
        sent = sent.replace('@@ ', '')
        if sent.endswith('@@'):
            sent = sent[:-len('@@')]
        return sent

    refs = list(map(lambda sents: list(map(join_and_decode, sents)), refs))
    hypos = list(map(join_and_decode, hypos))
    return tx.evals.corpus_bleu_moses(refs, hypos, return_all=False)


def build_model(batch, train_data, learning_rate):
    """Assembles the transformer model.
    """
    def single_bleu(ref, hypo):
        id2str = '<{}>'.format
        bos, eos = map(id2str, (bos_token_id, eos_token_id))

        ref = [id2str(u if u != unk_token_id else -1) for u in ref]
        hypo = [id2str(u) for u in hypo]

        ref = tx.utils.strip_special_tokens(
            ' '.join(ref), strip_bos=bos, strip_eos=eos)
        hypo = tx.utils.strip_special_tokens(
            ' '.join(hypo), strip_eos=eos)

        return 0.01 * tx.evals.sentence_bleu(references=[ref], hypothesis=hypo)


    def batch_bleu(refs, hypos):
        return np.array(
            [single_bleu(ref, hypo) for ref, hypo in zip(refs, hypos)],
            dtype=np.float32)


    train_ops = {}

    embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=config_model.embedder)

    source_embedder = embedder

    encoder = tx.modules.TransformerEncoder(
        hparams=config_model.encoder)

    enc_outputs = encoder(
        inputs=source_embedder(batch['source_text_ids']),
        sequence_length=batch['source_length'])

    target_embedder = embedder
    assert pad_token_id == 0
    target_embedding = tf.concat(
        [tf.zeros_like(embedder.embedding[:1]), embedder.embedding[1:]], axis=0)

    decoder = tx.modules.TransformerDecoder(
        embedding=target_embedding,
        hparams=config_model.decoder)

    # cross-entropy + teacher-forcing pretraining
    tf_outputs = decoder(
        decoding_strategy='train_greedy',
        memory=enc_outputs,
        memory_sequence_length=batch['source_length'],
        inputs=target_embedder(batch['target_text_ids']),
        sequence_length=batch['target_length'])

    loss_xe = transformer_utils.smoothing_cross_entropy(
        tf_outputs.logits,
        batch['labels'],
        vocab_size,
        config_train.loss_label_confidence)
    is_target = tf.to_float(tf.not_equal(batch['labels'], pad_token_id))
    loss_xe = tf.reduce_sum(loss_xe * is_target) / tf.reduce_sum(is_target)

    for xe_name in xe_names:
        train_ops[xe_name] = tx.core.get_train_op(
            loss_xe,
            learning_rate=learning_rate,
            hparams=getattr(config_train, 'train_{}'.format(xe_name)))

    # teacher mask + DEBLEU fine-tuning
    n_unmask = tf.placeholder(tf.int32, shape=[], name="n_unmask")
    n_mask = tf.placeholder(tf.int32, shape=[], name="n_mask")
    tm_helper = tx.modules.TeacherMaskSoftmaxEmbeddingHelper(
        # must not remove last token, since it may be used as mask
        inputs=batch['target_text_ids'],
        sequence_length=batch['target_length']-1,
        embedding=target_embedder,
        n_unmask=n_unmask,
        n_mask=n_mask,
        tau=config_train.tau)

    tm_outputs, _ = decoder(
        helper=tm_helper,
        memory=enc_outputs,
        memory_sequence_length=batch['source_length'],
        max_decoding_length=tf.shape(batch['target_text_ids'])[1])

    loss_debleu = tx.losses.debleu(
        labels=batch['target_text_ids'][:, 1:],
        probs=tm_outputs.sample_id,
        sequence_length=batch['target_length']-1,
        max_order=config_train.max_order,
        weights=config_train.weights)

    for debleu_name in debleu_names:
        train_ops[debleu_name] = tx.core.get_train_op(
            loss_debleu,
            hparams=getattr(config_train, 'train_{}'.format(debleu_name)))

    # start and end tokens
    start_tokens = tf.ones_like(batch['target_length']) * bos_token_id
    end_token = eos_token_id

    # inference: beam search decoding
    _bs_outputs = []
    for max_decoding_length, beam_width, alpha in config_train.infer_configs:
        bs_outputs = decoder(
            start_tokens=start_tokens,
            end_token=end_token,
            memory=enc_outputs,
            memory_sequence_length=batch['source_length'],
            beam_width=beam_width,
            alpha=alpha,
            max_decoding_length=max_decoding_length)
        bs_outputs = bs_outputs['sample_id']

        _bs_outputs.append(bs_outputs)

    bs_outputs = _bs_outputs

    # sampling:
    n_samples = config_train.n_samples

    def tile(a):
        return tf.tile(
            a, tf.concat([[n_samples], tf.ones_like(tf.shape(a)[1:])], -1))

    def untile(a):
        return tf.reshape(a, tf.concat([[n_samples, -1], tf.shape(a)[1:]], -1))

    _sample_outputs, _sample_length = [], []
    for i_sample in range(n_samples):
        sample_outputs, sample_length = decoder(
            decoding_strategy='infer_sample',
            start_tokens=start_tokens,
            end_token=end_token,
            memory=enc_outputs,
            memory_sequence_length=batch['source_length'],
            max_decoding_length=config_train.sample_max_decoding_length)
        _sample_outputs.append(sample_outputs)
        _sample_length.append(sample_length)

    def concat(*tensors):
        lengths = [tf.shape(tensor)[1] for tensor in tensors]
        max_length = tf.reduce_max(tf.stack(lengths))
        tensors = [
            tf.concat([
                tensor,
                tf.zeros(
                    tf.shape(tensor) + tf.one_hot(
                        1,
                        tf.shape(tf.shape(tensor))[0],
                        max_length - 2 * length),
                    dtype=tensor.dtype)],
                axis=1)
            for length, tensor in zip(lengths, tensors)]
        return tf.concat(tensors, axis=0)

    sample_outputs = tf.contrib.framework.nest.map_structure(
        concat, *_sample_outputs)
    sample_length = tf.concat(_sample_length, axis=0)

    sample_reward = tf.py_func(
        batch_bleu, [tile(batch['target_text_ids']), sample_outputs.sample_id],
        tf.float32, stateful=False, name='sample_reward')
    mean_sample_reward = tf.reduce_mean(
        untile(sample_reward), axis=0, name="sample_mean_reward")

    # greedy decoding:
    greedy_outputs, greedy_length = decoder(
        decoding_strategy='infer_greedy',
        start_tokens=start_tokens,
        end_token=end_token,
        memory=enc_outputs,
        memory_sequence_length=batch['source_length'],
        max_decoding_length=config_train.greedy_max_decoding_length)

    greedy_reward = tf.py_func(
        batch_bleu, [batch['target_text_ids'], greedy_outputs.sample_id],
        tf.float32, stateful=False, name='greedy_reward')

    # policy gradient
    nll_pg = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=sample_outputs.sample_id,
        logits=sample_outputs.logits,
        sequence_length=sample_length,
        average_across_batch=False)

    def get_pg_loss(baseline_reward, name):
        loss_pg = tf.reduce_mean(
            (sample_reward - tile(baseline_reward)) * nll_pg)
        weight_pg = getattr(config_train, 'weight_{}'.format(name))
        loss_pg_xe = loss_pg * weight_pg + loss_xe * (1. - weight_pg)
        train_ops[name] = tx.core.get_train_op(
            loss_pg_xe,
            hparams=getattr(config_train, 'train_{}'.format(name)))
        return loss_pg, loss_pg_xe, train_ops[name]

    # policy gradient with greedy baseline
    loss_pg_grd, loss_pg_grd_xe, _ = get_pg_loss(
        greedy_reward, pg_names[0])
    # policy gradient with mean sample baseline
    loss_pg_msp, loss_pg_msp_xe, _ = get_pg_loss(
        mean_sample_reward, pg_names[1])

    return train_ops, tm_helper, (n_unmask, n_mask), bs_outputs, \
        sample_outputs, loss_debleu


def main():
    """Entrypoint.
    """
    train_data, val_data, test_data = data_utils.load_data_numpy(
        config_data.input_dir, config_data.filename_prefix)
    dataset = {'train': train_data, 'val': val_data, 'test': test_data}
    with open(config_data.vocab_file, 'rb') as f:
        id2w = pickle.load(f)
    global vocab_size
    vocab_size = len(id2w)

    source_text_ids = tf.placeholder(tf.int64, shape=(None, None))
    target_text_ids = tf.placeholder(tf.int64, shape=(None, None))
    source_length = tf.reduce_sum(
        tf.to_int32(tf.not_equal(source_text_ids, pad_token_id)), axis=1)
    target_length = tf.reduce_sum(
        tf.to_int32(tf.not_equal(target_text_ids, pad_token_id)), axis=1)
    labels = tf.placeholder(tf.int64, shape=(None, None))
    data_batch = {
        'source_text_ids': source_text_ids,
        'target_text_ids': target_text_ids,
        'source_length': source_length,
        'target_length': target_length,
        'labels': labels,
    }

    global_step = tf.train.create_global_step()
    learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')
    tf.summary.scalar('lr', learning_rate)

    train_ops, tm_helper, mask_pattern_, bs_outputs, sample_outputs, \
        loss_debleu = build_model(data_batch, train_data, learning_rate)

    def get_train_op_scope(name):
        return get_scope_by_name(train_ops[name])

    train_op_initializers = {
        name: tf.variables_initializer(
            tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=get_train_op_scope(name)),
            name='train_{}_op_initializer'.format(name))
        for name in all_names}

    summary_tm = [
        tf.summary.scalar('tm/n_unmask', tm_helper.n_unmask),
        tf.summary.scalar('tm/n_mask', tm_helper.n_mask)]
    summary_ops = {
        name: tf.summary.merge(
            tf.get_collection(
                tf.GraphKeys.SUMMARIES,
                scope=get_train_op_scope(name))
            + (summary_tm if name in debleu_names else []),
            name='summary_{}'.format(name))
        for name in all_names}

    global convergence_trigger
    convergence_trigger = tx.utils.BestEverConvergenceTrigger(
        None,
        lambda state: state,
        config_train.threshold_steps,
        config_train.minimum_interval_steps)

    _saver = tf.train.Saver(max_to_keep=None)

    def _save_to(directory, step):
        print('saving to {} ...'.format(directory))
        saved_path = _saver.save(sess, directory, global_step=step)

        for trigger_name in ['convergence_trigger', 'annealing_trigger']:
            trigger = globals()[trigger_name]
            trigger_path = '{}.{}'.format(saved_path, trigger_name)
            print('saving {} ...'.format(trigger_name))
            with open(trigger_path, 'wb') as pickle_file:
                trigger.save_to_pickle(pickle_file)

        print('saved to {}'.format(saved_path))

    def _restore_from_path(ckpt_path, restore_trigger_names=None, relax=False):
        print('restoring from {} ...'.format(ckpt_path))
        (get_optimistic_saver(ckpt_path) if relax else _saver)\
            .restore(sess, ckpt_path)

        if restore_trigger_names is None:
            restore_trigger_names = ['convergence_trigger', 'annealing_trigger']

        for trigger_name in restore_trigger_names:
            trigger = globals()[trigger_name]
            trigger_path = '{}.{}'.format(ckpt_path, trigger_name)
            if os.path.exists(trigger_path):
                print('restoring {} ...'.format(trigger_name))
                with open(trigger_path, 'rb') as pickle_file:
                    trigger.restore_from_pickle(pickle_file)
            else:
                print('cannot find previous {} state.'.format(trigger_name))

        print('done.')

    def _restore_from(directory, restore_trigger_names=None, relax=False):
        if os.path.exists(directory):
            ckpt_path = tf.train.latest_checkpoint(directory)
            _restore_from_path(ckpt_path, restore_trigger_names, relax)

        else:
            print('cannot find checkpoint directory {}'.format(directory))

    def _train_epoch(sess, summary_writer, mode, train_op, summary_op):
        print('in _train_epoch')

        data = dataset[mode]
        random.shuffle(data)
        batches = torchtext.data.iterator.pool(
            data,
            config_data.n_tokens,
            key=lambda x: (len(x[0]), len(x[1])),
            batch_size_fn=utils.batch_size_fn,
            random_shuffler=torchtext.data.iterator.RandomShuffler())

        for batch in batches:
            step = sess.run(global_step)

            padded_batch = data_utils.seq2seq_pad_concat_convert(batch)
            feed_dict = {
                data_batch['source_text_ids']: padded_batch[0],
                data_batch['target_text_ids']: padded_batch[1],
                data_batch['labels']: padded_batch[2],
                learning_rate: utils.get_lr(step, config_train.lr),
                tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            }
            if mask_pattern is not None:
                feed_dict.update(
                    {mask_pattern_[_]: mask_pattern[_] for _ in range(2)})

            loss, summary, step = sess.run(
                (train_op, summary_op, global_step), feed_dict)

            summary_writer.add_summary(summary, step)

            if (step + 1) % config_train.steps_per_val == 0:
                global triggered
                _eval_epoch(sess, summary_writer, 'val')
                if triggered:
                    break

            if (step + 1) % config_train.steps_per_test == 0:
                _eval_epoch(sess, summary_writer, 'test')

        print('end _train_epoch')

    def _eval_epoch(sess, summary_writer, mode):
        print('in _eval_epoch with mode {}'.format(mode))

        data = dataset[mode]

        def _batches(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i : i + batch_size]

        batches = _batches(data, config_data.test_batch_size)

        texts = []
        fetches = bs_outputs

        if need_pickle:
            pickle_file = open('{}{}.pkl'.format(pickle_prefix, mode), 'wb')
            fetches += sample_outputs.sample_id, loss_debleu

        cnt = 0
        for batch in batches:
            sources, targets = zip(*batch)
            padded_batch = data_utils.seq2seq_pad_concat_convert(batch)
            feed_dict = {
                data_batch['source_text_ids']: padded_batch[0],
                data_batch['target_text_ids']: padded_batch[1],
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
                mask_pattern_[0]: 1,
                mask_pattern_[1]: 0,
            }

            fetched = sess.run(fetches, feed_dict)
            bs_output_ids = fetched[: len(bs_outputs)]
            bs_output_ids = list(map(lambda ids: ids[:, :, 0], bs_output_ids))
            all_ids = [targets] + bs_output_ids
            if need_pickle:
                sample_output_ids, _loss_debleu = fetched[-2:]
                all_ids += [sample_output_ids]

            all_ids = map(
                lambda sents: list(map(lambda sent: sent.tolist(), sents)),
                all_ids)
            all_texts = list(map(
                lambda ids: list(map(
                    lambda sent_ids: list(map(
                        lambda token_id: id2w[token_id],
                        strip_eos_id(sent_ids))),
                    ids)),
                all_ids))

            if need_pickle:
                pickle.dump(
                    all_texts + (_loss_debleu,),
                    pickle_file)

            texts.extend(zip(*all_texts[: 1 + len(bs_outputs)]))
            cnt += len(all_texts[0])
            print(cnt)

        if need_pickle:
            pickle_file.close()

        texts = list(zip(*texts))
        refs, hypos = texts[0], texts[1:]
        refs = list(map(lambda x: [x], refs))

        step = tf.train.global_step(sess, global_step)

        eval_bleu = None

        for i, hyps in enumerate(hypos):
            bleu = corpus_bleu(refs, hyps)
            print('{}_{} BLEU: {}'.format(mode, i, bleu))
            if eval_bleu is None:
                eval_bleu = bleu

            summary = tf.Summary()
            summary.value.add(tag='{}_{}/BLEU'.format(mode, i),
                              simple_value=bleu)
            summary_writer.add_summary(summary, step)
            summary_writer.flush()

        if mode == 'val':
            global triggered
            triggered = convergence_trigger(step, eval_bleu)
            if triggered:
                print('triggered!')

            if convergence_trigger.best_ever_step == step:
                print('updated best val bleu: {}'.format(
                    convergence_trigger.best_ever_score))

                _save_to(ckpt_best, step)

        _save_to(ckpt_model, step)

        print('end _eval_epoch')
        return eval_bleu

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        def action(i):
            if i >= len(phases) - 1:
                return i
            i += 1
            train_data_name, train_op_name, mask_pattern = phases[i]
            if reinitialize:
                sess.run(train_op_initializers[train_op_name])
            return i

        global annealing_trigger
        annealing_trigger = tx.utils.Trigger(0, action)

        def _restore_and_anneal():
            _restore_from(dir_best, ['convergence_trigger'])
            annealing_trigger.trigger()

        if restore_from:
            _restore_from_path(restore_from, relax=True)
        else:
            _restore_from(dir_model, relax=True)

        summary_writer = tf.summary.FileWriter(
            os.path.join(expr_name, 'log'), sess.graph, flush_secs=30)

        epoch = 0
        while epoch < config_train.max_epochs:
            train_data_name, train_op_name, mask_pattern = phases[
                annealing_trigger.user_state]
            train_op = train_ops[train_op_name]
            summary_op = summary_ops[train_op_name]

            print('epoch #{} {}:'.format(
                epoch, (train_data_name, train_op_name, mask_pattern)))

            val_bleu = _eval_epoch(sess, summary_writer, 'val')
            test_bleu = _eval_epoch(sess, summary_writer, 'test')
            if triggered:
                _restore_and_anneal()
                continue

            step = tf.train.global_step(sess, global_step)

            print('epoch: {}, step: {}, val BLEU: {}, test BLEU: {}'.format(
                epoch, step, val_bleu, test_bleu))

            _train_epoch(sess, summary_writer, train_data_name,
                         train_op, summary_op)
            if triggered:
                _restore_and_anneal()
                continue

            epoch += 1

            step = tf.train.global_step(sess, global_step)
            _save_to(ckpt_model, step)

        test_bleu = _eval_epoch(sess, summary_writer, 'test')
        print('epoch: {}, test BLEU: {}'.format(epoch, test_bleu))


if __name__ == '__main__':
    main()

