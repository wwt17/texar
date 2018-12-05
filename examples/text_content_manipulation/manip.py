"""
Text Content Manipulation
3-gated copy net.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import importlib
import tensorflow as tf
import texar as tx
from copy_net import CopyNetWrapper

flags = tf.flags
flags.DEFINE_string("config", "config_nba", "The config to use.")
FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)


def _main(_):
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config.data_hparams.items()}
    train_data = datasets['train']
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()
    batch_size, num_steps = [
        tf.shape(data_batch["value_text_ids"])[d] for d in range(2)]

    sent_vocab = train_data.vocab('sent')
    sent_embedder = tx.modules.WordEmbedder(
        vocab_size=sent_vocab.size, hparams=config.emb_hparams)
    entry_vocab = train_data.vocab('entry')
    entry_embedder = tx.modules.WordEmbedder(
        vocab_size=entry_vocab.size, hparams=config.structured_emb_hparams)
    attribute_vocab = train_data.vocab('attribute')
    attribute_embedder = tx.modules.WordEmbedder(
        vocab_size=attribute_vocab.size, hparams=config.structured_emb_hparams)
    value_vocab = train_data.vocab('entry')
    value_embedder = tx.modules.WordEmbedder(
        vocab_size=value_vocab.size, hparams=config.structured_emb_hparams)

    sent_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config.encoder_hparams)
    structured_data_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config.encoder_hparams)

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
        axis=2)  # [batch_size, tup_nums, hidden_size]
    sent_enc_outputs, _ = sent_encoder(sent_embeds)
    strutctured_data_enc_outputs, _ = structured_data_encoder(
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
        axis=2)
    tplt_sent_enc_outputs, _ = sent_encoder(tplt_sent_embeds)
    tplt_strutctured_data_enc_outputs, _ = structured_data_encoder(
        tplt_structured_data_embeds)

    # copy net
    cell = tx.core.layers.get_rnn_cell(config.rnn_cell_hparams)
    copy_net_cell = CopyNetWrapper(
        cell=cell,
        template_encoder_states=tf.concat(tplt_sent_enc_outputs, -1),
        template_encoder_input_ids=sents,
        structured_data_encoder_states=tf.concat(strutctured_data_enc_outputs, -1),
        structured_data_encoder_input_ids=tf.concat([entries, attributes, values], axis=1),
        vocab_size=sent_vocab.size)
    decoder = tx.modules.BasicRNNDecoder(
        cell=copy_net_cell,
        vocab_size=sent_vocab.size,
        hparams={"max_decoding_length_infer": config.max_num_steps + 2})
    initial_state = decoder.zero_state(batch_size=batch_size, dtype=tf.float32)
    outputs, _, _ = decoder(
        initial_state=initial_state,
        decoding_strategy="train_greedy",
        inputs=sent_embeds,
        sequence_length=data_batch["sent_length"] - 1)


if __name__ == '__main__':
    tf.app.run(main=_main)
