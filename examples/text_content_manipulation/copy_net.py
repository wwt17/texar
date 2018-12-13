import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util
import texar as tx


class CopyNetWrapperState(collections.namedtuple(
    "CopyNetWrapperState", ("cell_state", "time", "last_ids", "prob_tplt", "prob_sd"))):

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(CopyNetWrapperState, self)._replace(**kwargs))


class CopyNetWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(
            self, cell,
            tplt_encoder_states, tplt_encoder_input_ids,
            sd_encoder_states, sd_encoder_input_ids,
            vocab_size, input_ids, initial_cell_state=None,
            reuse=tf.AUTO_REUSE, name=None):
        super(CopyNetWrapper, self).__init__(name=name)

        with tf.variable_scope("CopyNetWrapper", reuse=reuse):
            self._cell = cell
            self._vocab_size = vocab_size
            self._input_ids = input_ids

            self._tplt_encoder_input_ids = tplt_encoder_input_ids
            self._tplt_encoder_states = tplt_encoder_states  # refer to h in the paper
            self._sd_encoder_input_ids = sd_encoder_input_ids
            self._sd_encoder_states = sd_encoder_states

            self._initial_cell_state = initial_cell_state
            self._tplt_copy_states = tf.layers.dense(
                self._tplt_encoder_states,
                units=self._cell.output_size,
                activation=tf.nn.tanh,
                use_bias=False)
            self._sd_copy_states = tf.layers.dense(
                self._sd_encoder_states,
                units=self._cell.output_size,
                activation=tf.nn.tanh,
                use_bias=False)
            self._projection = tf.layers.Dense(self._vocab_size, use_bias=False)

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError(
                "Expected state to be instance of CopyNetWrapperState. "
                "Received type {} instead.".format(type(state)))
        last_ids = state.last_ids
        last_ids = tf.cond(
            tx.utils.is_train_mode(tx.global_mode()),
            lambda: self._input_ids[:, state.time],
            lambda: last_ids)
        cell_state = state.cell_state

        def _get_selective_read(encoder_input_ids, encoder_states, prob):
            int_mask = tf.cast(
                tf.equal(tf.expand_dims(last_ids, 1), encoder_input_ids),
                tf.int32)
            int_mask_sum = tf.reduce_sum(int_mask, axis=1)
            mask = tf.cast(int_mask, tf.float32)
            mask_sum = tf.cast(int_mask_sum, tf.float32)
            mask = tf.where(
                tf.equal(int_mask_sum, 0),
                mask,
                mask / tf.expand_dims(mask_sum, 1))
            rou = mask * tf.cast(prob, tf.float32)
            return tf.einsum("ijk,ij->ik", encoder_states, rou)

        tplt_selective_read = _get_selective_read(
            self._tplt_encoder_input_ids,
            self._tplt_encoder_states,
            state.prob_tplt)
        sd_selective_read = _get_selective_read(
            self._sd_encoder_input_ids,
            self._sd_encoder_states,
            state.prob_sd)
        inputs = tf.concat([inputs, tplt_selective_read, sd_selective_read], -1)  # y_(t-1)

        # generate mode
        outputs, cell_state = self._cell(inputs, cell_state, scope)
        generate_score = self._projection(outputs)  # [batch, gen_vocab_size]
        generate_score = tf.cast(generate_score, tf.float64)
        exp_generate_score = tf.exp(generate_score)
        sumexp_generate_score = tf.reduce_sum(exp_generate_score, 1)

        # copy from template
        tplt_copy_score = tf.einsum(
            "ijm,im->ij",
            self._tplt_copy_states,
            outputs)  # [batch, num_steps]
        tplt_copy_score = tf.cast(tplt_copy_score, tf.float64)
        exp_tplt_copy_score = tf.exp(tplt_copy_score)
        sumexp_tplt_copy_score = tf.reduce_sum(exp_tplt_copy_score, 1)

        # copy from structured data
        sd_copy_score = tf.einsum(
            "ijm,im->ij",
            self._sd_copy_states,
            outputs)  # [batch, num_steps]
        sd_copy_score = tf.cast(sd_copy_score, tf.float64)
        exp_sd_copy_score = tf.exp(sd_copy_score)
        sumexp_sd_copy_score = tf.reduce_sum(exp_sd_copy_score, 1)

        Z = sumexp_generate_score + sumexp_tplt_copy_score + sumexp_sd_copy_score
        Z_ = tf.expand_dims(Z, 1)

        probs_generate = exp_generate_score / Z_


        def steps_to_vocabs(encoder_input_ids, prob):
            shape_of_encoder_input_ids = tf.shape(encoder_input_ids)
            batch_size = shape_of_encoder_input_ids[0]
            indices = tf.stack(
                [tf.tile(tf.expand_dims(tf.range(tf.cast(batch_size, tf.int64),
                                                 dtype=tf.int64),
                                        axis=-1),
                         [1, shape_of_encoder_input_ids[1]]),
                 encoder_input_ids],
                axis=-1)
            return tf.scatter_nd(indices, prob, [batch_size, self._vocab_size])


        prob_tplt = exp_tplt_copy_score / Z_
        probs_tplt = steps_to_vocabs(self._tplt_encoder_input_ids, prob_tplt)

        prob_sd = exp_sd_copy_score / Z_
        probs_sd = steps_to_vocabs(self._sd_encoder_input_ids, prob_sd)

        probs = probs_generate + probs_tplt + probs_sd
        outputs = tf.log(probs)
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int64)
        state = CopyNetWrapperState(
            time=state.time+1,
            cell_state=cell_state, last_ids=last_ids,
            prob_tplt=prob_tplt, prob_sd=prob_sd)
        outputs = tf.cast(outputs, tf.float32)
        return outputs, state

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
            It can be represented by an Integer, a TensorShape or a tuple of
            Integers or TensorShapes.
        """
        return CopyNetWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            last_ids=tf.TensorShape([]),
            prob_tplt=tf.shape(self._tplt_encoder_input_ids)[1],
            prob_sd=tf.shape(self._sd_encoder_input_ids)[1])

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._vocab_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState",
                           values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            last_ids = tf.zeros([batch_size], tf.int64) - 1
            prob_tplt = tf.zeros(
                [batch_size, tf.shape(self._tplt_encoder_states)[1]],
                tf.float64)
            prob_sd = tf.zeros(
                [batch_size, tf.shape(self._sd_encoder_states)[1]],
                tf.float64)
            return CopyNetWrapperState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int64), last_ids=last_ids,
                prob_tplt=prob_tplt, prob_sd=prob_sd)
