import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util


class CopyNetWrapperState(collections.namedtuple("CopyNetWrapperState",
                                                 ("cell_state", "last_ids", "prob_tplt", "prob_sd"))):

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
    def __init__(self, cell, template_encoder_states, template_encoder_input_ids,
                 structured_data_encoder_states, structured_data_encoder_input_ids,
                 vocab_size, encoder_state_size=None, initial_cell_state=None, name=None):
        super(CopyNetWrapper, self).__init__(name=name)
        self._cell = cell
        self._vocab_size = vocab_size

        self._template_encoder_input_ids = template_encoder_input_ids
        self._template_encoder_states = template_encoder_states  # refer to h in the paper
        self._structured_data_encoder_states = structured_data_encoder_states
        self._structured_data_encoder_input_ids = structured_data_encoder_input_ids

        if encoder_state_size is None:
            encoder_state_size = self._template_encoder_states.shape[-1].value
            if encoder_state_size is None:
                raise ValueError("encoder_state_size must be set if we can't "
                                 "infer encoder_states last dimension size.")
        self._encoder_state_size = encoder_state_size

        self._initial_cell_state = initial_cell_state
        self._template_copy_weight = \
            tf.get_variable('TemplateCopyWeight', [self._encoder_state_size, self._cell.output_size])
        self._structured_data_copy_weight = \
            tf.get_variable('StructuredDataCopyWeight', [self._encoder_state_size, self._cell.output_size])
        self._projection = tf.layers.Dense(self._vocab_size, use_bias=False, name="OutputProjection")

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of CopyNetWrapperState. "
                            "Received type %s instead." % type(state))
        last_ids = state.last_ids
        cell_state = state.cell_state

        def _get_selective_read(encoder_input_ids, encoder_states, prob):
            mask = tf.cast(tf.equal(tf.expand_dims(last_ids, 1), encoder_input_ids), tf.float32)
            mask_sum = tf.reduce_sum(mask, axis=1)
            mask = tf.where(tf.less(mask_sum, 1e-7), mask, mask / tf.expand_dims(mask_sum, 1))
            rou = mask * prob
            return tf.einsum("ijk,ij->ik", encoder_states, rou)

        tplt_selective_read = _get_selective_read(self._template_encoder_input_ids,
                                                  self._template_encoder_states, state.prob_tplt)
        sd_selective_read = _get_selective_read(self._structured_data_encoder_input_ids,
                                                self._structured_data_encoder_states, state.prob_sd)
        inputs = tf.concat([inputs, tplt_selective_read, sd_selective_read], 1)  # y_(t-1)

        # generate mode
        outputs, cell_state = self._cell(inputs, cell_state, scope)
        generate_score = self._projection(outputs)  # [batch, gen_vocab_size]

        # copy from template
        template_copy_score = tf.nn.tanh(
            tf.einsum("ijk,km->ijm", self._template_encoder_states, self._template_copy_weight))  # [batch, num_steps, m]
        template_copy_score = tf.einsum("ijm,im->ij", template_copy_score, outputs)  # [batch, num_steps]
        template_encoder_input_mask = tf.one_hot(self._template_encoder_input_ids, self._vocab_size)  # [batch, num_steps, vocab_size]
        expanded_template_copy_score = tf.einsum("ijn,ij->ij", template_encoder_input_mask, template_copy_score)  # [batch, num_steps]

        # copy from structured data
        structured_data_copy_score = tf.nn.tanh(
            tf.einsum("ijk,km->ijm", self._structured_data_encoder_states, self._structured_data_copy_weight))  # [batch, num_steps, m]
        structured_data_copy_score = tf.einsum("ijm,im->ij", structured_data_copy_score, outputs)  # [batch, num_steps]
        structured_data_encoder_input_mask = tf.one_hot(self._structured_data_encoder_input_ids, self._vocab_size)  # [batch, num_steps, vocab_size]
        expanded_structured_data_copy_score = \
            tf.einsum("ijn,ij->ij", structured_data_encoder_input_mask, structured_data_copy_score)  # [batch, num_steps]

        prob_g = generate_score
        prob_tplt = expanded_template_copy_score
        prob_sd = expanded_structured_data_copy_score

        prob_tplt_one_hot = tf.einsum("ijn,ij->in", template_encoder_input_mask, prob_tplt)  # [batch, vocab_size]
        prob_sd_one_hot = tf.einsum("ijn,ij->in", structured_data_encoder_input_mask, prob_sd)  # [batch, vocab_size]
        outputs = prob_g + prob_tplt_one_hot + prob_sd_one_hot
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int64)
        #prob_c.set_shape([None, self._encoder_state_size])
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_tplt=prob_tplt, prob_sd=prob_sd)
        return outputs, state

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(cell_state=self._cell.state_size, last_ids=tf.TensorShape([]),
                                   prob_tplt=self._encoder_state_size, prob_sd=self._encoder_state_size)

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._vocab_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            last_ids = tf.zeros([batch_size], tf.int64) - 1
            prob_tplt = tf.zeros([batch_size, tf.shape(self._template_encoder_states)[1]], tf.float32)
            prob_sd = tf.zeros([batch_size, tf.shape(self._template_encoder_states)[1]], tf.float32)
            return CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_tplt=prob_tplt, prob_sd=prob_sd)
