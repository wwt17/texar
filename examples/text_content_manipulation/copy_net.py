import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util
import texar as tx


class CopyNetWrapperState(collections.namedtuple(
    "CopyNetWrapperState", ("cell_state", "time", "last_ids", "copy_probs"))):

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
            self, cell, memory_ids_states_lengths, vocab_size,
            get_get_copy_scores, input_ids=None, initial_cell_state=None,
            reuse=tf.AUTO_REUSE, name=None):
        super(CopyNetWrapper, self).__init__(name=name)

        with tf.variable_scope("CopyNetWrapper", reuse=reuse):
            self._cell = cell
            self._memory_ids_states_lengths = [
                (memory_ids,
                 tx.utils.mask_sequences(
                    memory_states, memory_lengths, tensor_rank=3),
                 memory_lengths)
                for memory_ids, memory_states, memory_lengths in
                memory_ids_states_lengths]
            self._vocab_size = vocab_size
            self._input_ids = input_ids
            self._initial_cell_state = initial_cell_state
            self._get_copy_scores = get_get_copy_scores(
                memory_ids_states_lengths, self._cell.output_size)
            self._projection = tf.layers.Dense(self._vocab_size, use_bias=False)
            self._path_weights_projection = tf.layers.Dense(
                1 + len(self._memory_ids_states_lengths), use_bias=False)

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError(
                "Expected state to be instance of CopyNetWrapperState. "
                "Received type {} instead.".format(type(state)))
        last_ids = state.last_ids
        if self._input_ids is not None:
            last_ids = tf.cond(
                tx.utils.is_train_mode(tx.global_mode()),
                lambda: self._input_ids[:, state.time],
                lambda: last_ids)
        cell_state = state.cell_state

        def _get_selective_read(memory_ids, memory_states, memory_lengths, prob):
            int_mask = tf.cast(
                tf.equal(tf.expand_dims(last_ids, 1), memory_ids),
                tf.int32)
            int_mask_sum = tf.reduce_sum(int_mask, axis=1)
            mask = tf.cast(int_mask, tf.float32)
            mask_sum = tf.cast(int_mask_sum, tf.float32)
            mask = tf.where(
                tf.equal(int_mask_sum, 0),
                mask,
                mask / tf.expand_dims(mask_sum, 1))
            rou = mask * tf.cast(prob, tf.float32)
            return tf.einsum("ijk,ij->ik", memory_states, rou)

        inputs = [inputs]
        for (memory_ids, memory_states, memory_lengths), copy_prob in \
                zip(self._memory_ids_states_lengths, state.copy_probs):
            inputs.append(_get_selective_read(
                memory_ids, memory_states, memory_lengths, copy_prob))
        inputs = tf.concat(inputs, -1)  # y_(t-1)

        outputs, cell_state = self._cell(inputs, cell_state, scope)
        path_weights = tf.nn.softmax(self._path_weights_projection(outputs))
        path_weights = tf.cast(path_weights, tf.float64)

        # generate mode
        generate_score = self._projection(outputs)  # [batch, gen_vocab_size]
        generate_score = tf.cast(generate_score, tf.float64)
        exp_generate_score = tf.exp(generate_score) * path_weights[:, 0:1]
        sumexp_generate_score = tf.reduce_sum(exp_generate_score, 1)
        Z = sumexp_generate_score

        # copy from memory
        copy_scores = self._get_copy_scores(outputs)
        exp_copy_scores = [
            tx.utils.mask_sequences(
                tf.exp(tf.cast(copy_score, tf.float64)), memory_lengths)
            * path_weights[:, i+1:i+2]
            for i, ((_, _, memory_lengths), copy_score) in
            enumerate(zip(self._memory_ids_states_lengths, copy_scores))]
        for exp_copy_score in exp_copy_scores:
            sumexp_copy_score = tf.reduce_sum(exp_copy_score, 1)
            Z = Z + sumexp_copy_score

        Z_ = tf.expand_dims(Z, 1)

        probs_generate = exp_generate_score / Z_
        probs = probs_generate


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


        copy_probs = []
        for (memory_ids, _, _), exp_copy_score in zip(
                self._memory_ids_states_lengths, exp_copy_scores):
            copy_prob = exp_copy_score / Z_
            copy_probs.append(copy_prob)
            probs_copy = steps_to_vocabs(memory_ids, copy_prob)
            probs = probs + probs_copy

        outputs = tf.log(probs)
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int64)
        state = CopyNetWrapperState(
            cell_state=cell_state,
            time=state.time+1, last_ids=last_ids,
            copy_probs=copy_probs)
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
            copy_probs=[tf.shape(memory_ids)[1] for memory_ids, _, _ in
                       self._memory_ids_states_lengths],
        )

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
            copy_probs = [
                tf.zeros([batch_size, tf.shape(memory_ids)[1]], tf.float64)
                for memory_ids, _, _ in self._memory_ids_states_lengths]
            return CopyNetWrapperState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int64), last_ids=last_ids,
                copy_probs=copy_probs)
