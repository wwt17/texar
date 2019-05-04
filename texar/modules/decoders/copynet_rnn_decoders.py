# Copyright 2019 The Texar Authors. All Rights Reserved.
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
"""
CopyNet RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=not-context-manager, protected-access, invalid-name

import collections
import copy

import numpy as np

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.contrib.seq2seq import tile_batch

from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase
from texar.utils import utils
from texar.utils.dtypes import is_callable

__all__ = [
    "CopyingMechanism",
    "BasicCopyingMechanism",
    "CopyNetWrapperState",
    "CopyNetWrapper",
    "CopyNetRNNDecoderOutput",
    "CopyNetRNNDecoder"
]


def _prepare_memory(memory_id,
                    memory_state,
                    memory_sequence_length,
                    check_inner_dims_defined):
    """Convert to tensor and possibly mask `memory`.
    Args:
        memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
        memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
        check_inner_dims_defined: Python boolean.  If `True`, the `memory`
            argument's shape is checked to ensure all but the two outermost
            dimensions are fully defined.
    Returns:
        A (possibly masked), checked, new `memory`.
    Raises:
        ValueError: If `check_inner_dims_defined` is `True` and not
            `memory.shape[2:].is_fully_defined()`.
    """
    memory_id = nest.map_structure(
        lambda m: tf.convert_to_tensor(m, name="memory_id"), memory_id)
    memory_state = nest.map_structure(
        lambda m: tf.convert_to_tensor(m, name="memory_state"), memory_state)
    if memory_sequence_length is not None:
        memory_sequence_length = tf.convert_to_tensor(
            memory_sequence_length, name="memory_sequence_length")
    if check_inner_dims_defined:
        def _check_dims(m):
            if not m.get_shape()[2:].is_fully_defined():
                raise ValueError(
                    "Expected memory_state %s to have fully defined inner dims, "
                    "but saw shape: %s" % (m.name, m.get_shape()))
        nest.map_structure(_check_dims, memory_state)
    if memory_sequence_length is None:
        seq_len_mask = None
    else:
        seq_len_mask = tf.sequence_mask(
            memory_sequence_length,
            maxlen=tf.shape(nest.flatten(memory_state)[0])[1],
            dtype=nest.flatten(memory_state)[0].dtype)
        seq_len_batch_size = (
            memory_sequence_length.shape[0].value
            or tf.shape(memory_sequence_length)[0])
    def _maybe_mask(m, seq_len_mask):
        rank = m.get_shape().ndims
        rank = rank if rank is not None else tf.rank(m)
        extra_ones = tf.ones(rank - 2, dtype=tf.int32)
        m_batch_size = m.shape[0].value or tf.shape(m)[0]
        if memory_sequence_length is not None:
            message = ("memory_sequence_length and memory tensor batch sizes do not "
                       "match.")
            with tf.control_dependencies([
                    tf.assert_equal(
                        seq_len_batch_size, m_batch_size, message=message)]):
                seq_len_mask = tf.reshape(
                    seq_len_mask,
                    tf.concat((tf.shape(seq_len_mask), extra_ones), 0))
                return m * seq_len_mask
        else:
            return m
    return memory_id, \
        nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory_state)


def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
    if memory_sequence_length is None:
        return score
    message = ("All values in memory_sequence_length must greater than zero.")
    with tf.control_dependencies(
            [tf.assert_positive(memory_sequence_length, message=message)]):
        score_mask = tf.sequence_mask(
            memory_sequence_length, maxlen=tf.shape(score)[1])
        score_mask_values = score_mask_value * tf.ones_like(score)
        return tf.where(score_mask, score, score_mask_values)


class CopyingMechanism(object):
    """A base CopyingMechanism class providing common functionality.
    Common functionality includes:
        1. Storing the query and memory layers.
        2. Preprocessing and storing the memory.
    """

    def __init__(self,
                 memory_id,
                 memory_state,
                 memory_sequence_length=None,
                 memory_layer=None,
                 query_layer=None,
                 check_inner_dims_defined=True,
                 scope=None,
                 name=None):
        """Construct base CopyingMechanism class.
        Args:
        memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
        memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
        memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
            depth must match the depth of `query_layer`.
            If `memory_layer` is not provided, the shape of `memory` must match
            that of `query_layer`.
        query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
            must match the depth of `memory_layer`.  If `query_layer` is not
            provided, the shape of `query` must match that of `memory_layer`.
        check_inner_dims_defined: Python boolean.  If `True`, the `memory`
            argument's shape is checked to ensure all but the two outermost
            dimensions are fully defined.
        name: Name to use when creating ops.
        """
        self._memory_layer = memory_layer
        self._query_layer = query_layer
        self.dtype = memory_layer.dtype

        if scope is None:
            with tf.variable_scope(None, "basic_copying_mechanism") as scope:
                pass
        self._variable_scope = scope

        with tf.name_scope(
                name, "CopyingMechanismInit",
                nest.flatten(memory_id) + nest.flatten(memory_state)):
            with tf.variable_scope(self._variable_scope, reuse=tf.AUTO_REUSE):
                self._memory_id, self._memory_state = _prepare_memory(
                    memory_id, memory_state, memory_sequence_length,
                    check_inner_dims_defined=check_inner_dims_defined)
                self._memory_sequence_length = memory_sequence_length
                self._keys = (
                    self.memory_layer(self._memory_state) if self.memory_layer  # pylint: disable=not-callable
                    else self._memory_state)
                self._batch_size = (
                    self._keys.shape[0].value or
                    tf.shape(self._keys)[0])
                self._memory_size = (
                    self._keys.shape[1].value or
                    tf.shape(self._keys)[1])

    def __call__(self, query, coverity_state):
        raise NotImplementedError

    def map_to_vocab(self, copying_probability, vocab_size):
        raise NotImplementedError

    def initial_copying_probability(self, batch_size, dtype):
        raise NotImplementedError

    def initial_coverity_state(self, batch_size, dtype):
        raise NotImplementedError

    def update_coverity_state(self, coverity_state, copying_probability, cell_outputs):
        raise NotImplementedError

    def selective_read(self, last_ids, copying_probability):
        raise NotImplementedError

    def _get_beam_search_copying_mechanism(self):
        raise NotImplementedError

    @property
    def memory_layer(self):
        return self._memory_layer

    @property
    def query_layer(self):
        return self._query_layer

    @property
    def memory_id(self):
        return self._memory_id

    @property
    def memory_state(self):
        return self._memory_state

    @property
    def keys(self):
        return self._keys

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def memory_size(self):
        return self._memory_size

    @property
    def coverity_state_size(self):
        raise NotImplementedError

    @property
    def variable_scope(self):
        return self._variable_scope


class BasicCopyingMechanism(CopyingMechanism):

    def __init__(self,
                 num_units,
                 memory_id,
                 memory_state,
                 memory_sequence_length=None,
                 activation=tf.tanh,
                 coverity_state_dim=None,
                 coverity_cell=None,
                 score_mask_value=None,
                 scope=None,
                 name="BasicCopyingMechanism"):
        # For BasicCopyingMechanism, we only transform the memory layer; thus
        # num_units **must** match expected the query depth.
        super(BasicCopyingMechanism, self).__init__(
            memory_id=memory_id,
            memory_state=memory_state,
            memory_sequence_length=memory_sequence_length,
            memory_layer=tf.layers.Dense(
                num_units,
                activation=None,
                use_bias=False,
                name="memory_layer"),
            query_layer=None,
            scope=scope,
            name=name)
        self._num_units = num_units
        self._activation = activation
        self._coverity_state_dim = coverity_state_dim
        self._coverity_cell = coverity_cell
        self._name = name

        if score_mask_value is None:
            score_mask_value = -np.inf
        self._score_mask_value = score_mask_value

    def __call__(self, query, coverity_state):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
        state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
            score: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        with tf.variable_scope(
                self._variable_scope,
                values=[query, coverity_state],
                reuse=tf.AUTO_REUSE):
            keys = self._keys
            if coverity_state is not ():
                coverity_state = tf.reshape(
                    coverity_state,
                    [tf.shape(coverity_state)[0], -1,
                        self._coverity_state_dim])
                keys = keys + tf.layers.dense(
                    coverity_state,
                    units=self._num_units,
                    activation=None,
                    use_bias=False)
            keys = self._activation(keys)
            score = tf.einsum("bim,bm->bi", self._memory_state, query)
            score = _maybe_mask_score(
                score, self._memory_sequence_length, self._score_mask_value)
            return score

    def map_to_vocab(self, copying_probability, vocab_size):
        batch_size = self._batch_size
        dtype = self._memory_id.dtype
        indices = tf.stack(
            [tf.tile(tf.expand_dims(tf.range(tf.cast(batch_size, dtype),
                                             dtype=dtype),
                                    axis=-1),
                     [1, self._memory_size]),
             self._memory_id],
            axis=-1)
        return tf.scatter_nd(
            indices, copying_probability, [batch_size, vocab_size])

    def initial_copying_probability(self, batch_size, dtype):
        return tf.zeros([batch_size, self._memory_size], dtype=dtype,
                        name="initial_copying_probability")

    def initial_coverity_state(self, batch_size, dtype):
        return tf.zeros(
            [batch_size, self._memory_size * self._coverity_state_dim],
            dtype=dtype, name="initial_coverity_state")

    def update_coverity_state(self, coverity_state, copying_probability, state):
        coverity_state_shape = tf.shape(coverity_state)
        shape = tf.shape(copying_probability)
        state = tf.broadcast_to(
            tf.expand_dims(state, 1),
            tf.concat([shape, [state.shape[-1]]], -1))
        coverity_state = tf.reshape(
            coverity_state, [-1, self._coverity_state_dim])
        copying_probability = tf.reshape(copying_probability, [-1, 1])
        state = tf.reshape(state, [-1, state.shape[-1]])
        _, coverity_state = self._coverity_cell(
            tf.concat([copying_probability, state], -1), coverity_state)
        coverity_state = tf.reshape(coverity_state, coverity_state_shape)
        return coverity_state

    def selective_read(self, last_ids, copying_probability):
        """This method implements Eq.(9).
        """
        dtype = copying_probability.dtype
        int_mask = tf.cast(
            tf.equal(tf.expand_dims(last_ids, 1), self._memory_id),
            tf.int32)
        int_sum_mask = tf.reduce_sum(int_mask, axis=1)
        mask = tf.cast(int_mask, dtype)
        sum_mask = tf.cast(int_sum_mask, dtype)
        mask = tf.where(
            tf.equal(int_sum_mask, 0),
            mask,
            mask / tf.expand_dims(sum_mask, 1))
        rho = mask * copying_probability
        return tf.einsum("ij,ijk->ik", rho, self._memory_state)

    def _get_beam_search_copying_mechanism(self, beam_width):
        return BasicCopyingMechanism(
            self._num_units,
            tile_batch(self._memory_id, beam_width),
            tile_batch(self._memory_state, beam_width),
            memory_sequence_length=
                None if self._memory_sequence_length is None else
                tile_batch(self._memory_sequence_length, beam_width),
            activation=self._activation,
            coverity_state_dim=self._coverity_state_dim,
            coverity_cell=self._coverity_cell,
            scope=self._variable_scope,
            name="BeamSearch{}".format(self._name))

    @property
    def coverity_state_size(self):
        return self._memory_size * self._coverity_state_dim


class CopyNetWrapperState(
    collections.namedtuple("CopyNetWrapperState",
                           ("cell_state", "time", "copying_probability",
                            "copying_probability_history", "coverity_state"))):
    """`namedtuple` storing the state of a `CopyNetWrapper`.
    Contains:
        - `cell_state`: The state of the wrapped `RNNCell` at the previous time
            step.
        - `attention`: The attention emitted at the previous time step.
        - `time`: int32 scalar containing the current time step.
        - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
            emitted at the previous time step for each attention mechanism.
        - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
            containing alignment matrices from all time steps for each attention
            mechanism. Call `stack()` on each to convert to a `Tensor`.
        - `attention_state`: A single or tuple of nested objects
            containing attention mechanism state for each attention mechanism.
            The objects may contain Tensors or TensorArrays.
    """

    def clone(self, **kwargs):
        """Clone this object, overriding components provided by kwargs.
        The new state fields' shape must match original state fields' shape. This
        will be validated, and original fields' shape will be propagated to new
        fields.
        Example:
        ```python
        initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
        initial_state = initial_state.clone(cell_state=encoder_state)
        ```
        Args:
            **kwargs: Any properties of the state object to replace in the returned
                `CopyNetWrapperState`.
        Returns:
            A new `CopyNetWrapperState` whose properties are the same as
            this one, except any overridden properties as provided in `kwargs`.
        """
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
    """Wraps another `RNNCell` with CopyNet.
    """

    def __init__(self,
                 cell,
                 copying_mechanism,
                 vocab_size,
                 copying_probability_history=False,
                 coverity_state=True,
                 initial_cell_state=None,
                 generating_layer=None,
                 name=None):

        super(CopyNetWrapper, self).__init__(name=name)

        if isinstance(copying_mechanism, (list, tuple)):
            self._is_multi = True
            copying_mechanisms = copying_mechanism
            for copying_mechanism in copying_mechanisms:
                if not isinstance(copying_mechanism, CopyingMechanism):
                    raise TypeError(
                        "copying_mechanism must contain only instances of "
                        "CopyingMechanism, saw type: %s"
                        % type(copying_mechanism).__name__)
        else:
            self._is_multi = False
            if not isinstance(copying_mechanism, CopyingMechanism):
                raise TypeError(
                    "copying_mechanism must be an CopyingMechanism or list of "
                    "multiple CopyingMechanism instances, saw type: %s"
                    % type(copying_mechanism).__name__)
            copying_mechanisms = (copying_mechanism,)

        self._vocab_size = vocab_size
        if generating_layer is None:
            generating_layer = tf.layers.Dense(
                self._vocab_size, use_bias=False)
        self._generating_layer = generating_layer

        self._cell = cell
        self._copying_mechanism = copying_mechanism
        self._copying_mechanisms = copying_mechanisms
        self._copying_probability_history = copying_probability_history
        self._coverity_state = coverity_state
        with tf.name_scope(name, "CopyNetWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                    final_state_tensor.shape[0].value
                    or tf.shape(final_state_tensor)[0])
                error_message = (
                    "When constructing CopyNetWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and initial_cell_state.  Are you using "
                    "the BeamSearchDecoder?  You may need to tile your initial state "
                    "via the tf.contrib.seq2seq.tile_batch function with argument "
                    "multiple=beam_width.")
                with tf.control_dependencies(
                        self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = nest.map_structure(
                        lambda s: array_ops.identity(s, name="check_initial_cell_state"),
                        initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [tf.assert_equal(batch_size,
                                copying_mechanism.batch_size,
                                message=error_message)
                for copying_mechanism in self._copying_mechanisms]

    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the CopyingMechanism(s) were passed
        to the constructor.
        Args:
            seq: A non-empty sequence of items or generator.
        Returns:
            Either the values in the sequence as a tuple if CopyingMechanism(s)
            were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]

    @property
    def copying_mechanism(self):
        return self._copying_mechanism

    @property
    def generating_layer(self):
        return self._generating_layer

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def output_size(self):
        return self._vocab_size

    @property
    def state_size(self):
        """The `state_size` property of `CopyNetWrapper`.
        Returns:
            An `CopyNetWrapperState` tuple containing shapes used by this object.
        """
        return CopyNetWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            copying_probability=self._item_or_tuple(
                c.memory_size for c in self._copying_mechanisms),
            copying_probability_history=self._item_or_tuple(
                c.memory_size if self._copying_probability_history else ()
                for c in self._copying_mechanisms),  # sometimes a TensorArray
            coverity_state=self._item_or_tuple(
                c.coverity_state_size if self._coverity_state else ()
                for c in self._copying_mechanisms))

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `CopyNetWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `CopyNetWrapper` with a
        `BeamSearchDecoder`.
        Args:
            batch_size: `0D` integer tensor: the batch size.
            dtype: The internal state data type.
        Returns:
            An `CopyNetWrapperState` tuple containing zeroed out tensors and,
            possibly, empty `TensorArray` objects.
        Raises:
            ValueError: (or, possibly at runtime, InvalidArgument), if
                `batch_size` does not match the output size of the encoder passed
                to the wrapper object at initialization time.
        """
        with tf.name_scope(type(self).__name__ + "ZeroState",
                           values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                "When calling zero_state of CopyNetWrapper %s: " % self._base_name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the requested batch size.  Are you using "
                "the BeamSearchDecoder?  If so, make sure your encoder output has "
                "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                "the batch_size= argument passed to zero_state is "
                "batch_size * beam_width.")
            with tf.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: tf.identity(s, name="checked_cell_state"),
                    cell_state)
            initial_copying_probabilities = [
                copying_mechanism.initial_copying_probability(batch_size, dtype)
                for copying_mechanism in self._copying_mechanisms]
            return CopyNetWrapperState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int32),
                copying_probability=self._item_or_tuple(
                    initial_copying_probabilities),
                copying_probability_history=self._item_or_tuple(
                    tf.TensorArray(
                        dtype,
                        size=0,
                        dynamic_size=True,
                        element_shape=copying_probability.shape)
                    if self._copying_probability_history else ()
                    for copying_probability in initial_copying_probabilities),
                coverity_state=self._item_or_tuple(
                    copying_mechanism.initial_coverity_state(batch_size, dtype)
                    if self._coverity_state else ()
                    for copying_mechanism in self._copying_mechanisms))

    def call(self, inputs, state):
        if not isinstance(state, CopyNetWrapperState):
          raise TypeError("Expected state to be instance of CopyNetWrapperState. "
                          "Received type {} instead.".format(type(state)))

        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or
            tf.shape(cell_output)[0])
        error_message = (
            "When applying CopyNetWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                cell_output, name="checked_cell_output")

        # generating mode
        generating_score = self._generating_layer(cell_output)
        max_generating_score = tf.reduce_max(generating_score, axis=1, keepdims=True)
        max_score = max_generating_score

        # copying mode
        if self._is_multi:
            previous_coverity_state = state.coverity_state
            previous_copying_probability_history = \
                state.copying_probability_history
        else:
            previous_coverity_state = [state.coverity_state]
            previous_copying_probability_history = \
                [state.copying_probability_history]

        all_copying_scores = []
        for i, copying_mechanism in enumerate(self._copying_mechanisms):
            copying_score = copying_mechanism(
                cell_output, coverity_state=previous_coverity_state[i])
            max_copying_score = tf.reduce_max(copying_score, axis=1, keepdims=True)
            max_score = tf.maximum(max_score, max_copying_score)

            all_copying_scores.append(copying_score)

        # in order to avoid float overflow in exp(score)
        generating_score = generating_score - max_score
        all_copying_scores = [copying_score - max_score
                              for copying_score in all_copying_scores]

        # get exp(score)s
        exp_generating_score = tf.exp(generating_score)
        all_exp_copying_scores = list(map(tf.exp, all_copying_scores))

        # get sum of all exp(score)s (i.e. Z, the normalization term)
        sum_exp_generating_score = tf.reduce_sum(
            exp_generating_score, axis=1, keepdims=True)
        sum_exp_score = sum_exp_generating_score
        for exp_copying_score in all_exp_copying_scores:
            sum_exp_copying_score = tf.reduce_sum(
                exp_copying_score, axis=1, keepdims=True)
            sum_exp_score = sum_exp_score + sum_exp_copying_score

        generating_probability = exp_generating_score / sum_exp_score
        output_probability = generating_probability
        all_copying_probabilities = []
        all_next_coverity_states = []
        maybe_all_histories = []
        for i, (copying_mechanism, exp_copying_score) in enumerate(zip(
                self._copying_mechanisms, all_exp_copying_scores)):
            copying_probability = exp_copying_score / sum_exp_score
            all_copying_probabilities.append(copying_probability)
            next_coverity_state = copying_mechanism.update_coverity_state(
                    previous_coverity_state[i], copying_probability,
                    cell_output) \
                    if self._coverity_state is not None else ()
            copying_probability_history = \
                previous_copying_probability_history[i].write(
                    state.time, copying_probability) \
                if self._copying_probability_history else ()

            all_next_coverity_states.append(next_coverity_state)
            maybe_all_histories.append(copying_probability_history)
            copying_probability_over_vocab = copying_mechanism.map_to_vocab(
                copying_probability, self._vocab_size)
            output_probability = output_probability + \
                                 copying_probability_over_vocab

        next_state = CopyNetWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            copying_probability=self._item_or_tuple(all_copying_probabilities),
            copying_probability_history=\
                self._item_or_tuple(maybe_all_histories),
            coverity_state=self._item_or_tuple(all_next_coverity_states))

        return output_probability, next_state


class CopyNetRNNDecoderOutput(
        collections.namedtuple(
            "CopyNetRNNDecoderOutput",
            ["logits", "sample_id", "cell_output", "copying_probability"])):
    """The outputs of CopyNet RNN decoders that additionally include
    copy results.

    Attributes:
        logits: The outputs of RNN (at each step/of all steps) by applying the
            output layer on cell outputs. E.g., in
            :class:`~texar.modules.AttentionRNNDecoder`, this is a Tensor of
            shape `[batch_size, max_time, vocab_size]` after decoding.
        sample_id: The sampled results (at each step/of all steps). E.g., in
            :class:`~texar.modules.AttentionRNNDecoder` with decoding strategy
            of train_greedy, this
            is a Tensor of shape `[batch_size, max_time]` containing the
            sampled token indexes of all steps.
        cell_output: The output of RNN cell (at each step/of all steps).
            This is the results prior to the output layer. E.g., in
            AttentionRNNDecoder with default
            hyperparameters, this is a Tensor of
            shape `[batch_size, max_time, cell_output_size]` after decoding
            the whole sequence.
        copying_probability: A single or tuple of `Tensor`(s) containing the
            alignments emitted (at the previous time step/of all time steps)
            for each copying mechanism.
    """
    pass


class CopyNetRNNDecoder(RNNDecoderBase):
    """RNN decoder with copying mechanism.

    Args:
        memory: The memory to query, e.g., the output of an RNN encoder. This
            tensor should be shaped `[batch_size, max_time, dim]`.
        memory_sequence_length (optional): A tensor of shape `[batch_size]`
            containing the sequence lengths for the batch
            entries in memory. If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
        cell (RNNCell, optional): An instance of `RNNCell`. If `None`, a cell
            is created as specified in :attr:`hparams`.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in
            TRAIN mode). If `None`, :func:`~texar.global_mode` is used.
            Ignored if :attr:`cell` is given.
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        output_layer (optional): An output layer that transforms cell output
            to logits. This can be:

            - A callable layer, e.g., an instance \
            of :tf_main:`tf.layers.Layer <layers/Layer>`.
            - A tensor. A dense layer will be created using the tensor \
            as the kernel weights. The bias of the dense layer is determined by\
            `hparams.output_layer_bias`. This can be used to tie the output \
            layer with the input embedding matrix, as proposed in \
            https://arxiv.org/pdf/1608.05859.pdf
            - `None`. A dense layer will be created based on attr:`vocab_size`\
            and `hparams.output_layer_bias`.
            - If no output layer after the cell output is needed, set \
            `(vocab_size=None, output_layer=tf.identity)`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`~texar.modules.RNNDecoderBase._build` for the inputs and outputs
    of the decoder. The decoder returns
    `(outputs, final_state, sequence_lengths)`, where `outputs` is an instance
    of :class:`~texar.modules.AttentionRNNDecoderOutput`.

    Example:

        .. code-block:: python

            # Encodes the source
            enc_embedder = WordEmbedder(data.source_vocab.size, ...)
            encoder = UnidirectionalRNNEncoder(...)

            enc_outputs, _ = encoder(
                inputs=enc_embedder(data_batch['source_text_ids']),
                sequence_length=data_batch['source_length'])

            # Decodes while attending to the source
            dec_embedder = WordEmbedder(vocab_size=data.target_vocab.size, ...)
            decoder = AttentionRNNDecoder(
                memory=enc_outputs,
                memory_sequence_length=data_batch['source_length'],
                vocab_size=data.target_vocab.size)

            outputs, _, _ = decoder(
                decoding_strategy='train_greedy',
                inputs=dec_embedder(data_batch['target_text_ids']),
                sequence_length=data_batch['target_length']-1)
    """
    def __init__(self,
                 copying_mechanism,
                 vocab_size,
                 cell=None,
                 cell_dropout_mode=None,
                 output_layer=tf.log,
                 initial_cell_state=None,
                 generating_layer=None,
                 selective_read_input_fn=None,
                 hparams=None):

        if not is_callable(output_layer):
            raise ValueError(
                "`output_layer` must be a callable, but {} is provided.".format(
                    output_layer))
        output_layer = tf.keras.layers.Lambda(output_layer)

        RNNDecoderBase.__init__(
            self, cell, vocab_size, output_layer, cell_dropout_mode, hparams)

        copy_hparams = self._hparams['copying']

        self._copy_cell_kwargs = {
            name: copy_hparams[name] for name in
            {"copying_probability_history", "coverity_state"}
        }
        self._initial_cell_state = initial_cell_state
        self._selective_read = copy_hparams["selective_read"]
        self._selective_read_input_fn = selective_read_input_fn
        # Use variable_scope to ensure all trainable variables created in
        # CopyNetWrapper are collected
        with tf.variable_scope(self.variable_scope):
            copy_cell = CopyNetWrapper(
                self._cell,
                copying_mechanism,
                self._vocab_size,
                initial_cell_state=self._initial_cell_state,
                generating_layer=generating_layer,
                **self._copy_cell_kwargs)
            self._cell = copy_cell

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:

        Common hyperparameters are the same as in
        :class:`~texar.modules.BasicRNNDecoder`.
        :meth:`~texar.modules.BasicRNNDecoder.default_hparams`.
        Additional hyperparameters are for attention mechanism
        configuration.

        .. code-block:: python

            {
                "copy": {
                    "type": "LuongAttention",
                    "kwargs": {
                        "num_units": 256,
                    },
                    "alignment_history": False,
                    "output_attention": True,
                },
                # The following hyperparameters are the same as with
                # `BasicRNNDecoder`
                "rnn_cell": default_rnn_cell_hparams(),
                "max_decoding_length_train": None,
                "max_decoding_length_infer": None,
                "helper_train": {
                    "type": "TrainingHelper",
                    "kwargs": {}
                }
                "helper_infer": {
                    "type": "SampleEmbeddingHelper",
                    "kwargs": {}
                }
                "name": "attention_rnn_decoder"
            }

        Here:

        "copy" : dict
            Attention hyperparameters, including:

            "type" : str or class or instance
                The attention type. Can be an attention class, its name or
                module path, or a class instance. The class must be a subclass
                of :tf_main:`TF CopyingMechanism
                <contrib/seq2seq/CopyingMechanism>`. If class name is
                given, the class must be from modules
                :tf_main:`tf.contrib.seq2seq <contrib/seq2seq>` or
                :mod:`texar.custom`.

                Example:

                    .. code-block:: python

                        # class name
                        "type": "LuongAttention"
                        "type": "BahdanauAttention"
                        # module path
                        "type": "tf.contrib.seq2seq.BahdanauMonotonicAttention"
                        "type": "my_module.MyCopyingMechanismClass"
                        # class
                        "type": tf.contrib.seq2seq.LuongMonotonicAttention
                        # instance
                        "type": LuongAttention(...)

            "kwargs" : dict
                keyword arguments for the attention class constructor.
                Arguments :attr:`memory` and
                :attr:`memory_sequence_length` should **not** be
                specified here because they are given to the decoder
                constructor. Ignored if "type" is an attention class
                instance. For example

                Example:

                    .. code-block:: python

                        "type": "LuongAttention",
                        "kwargs": {
                            "num_units": 256,
                            "probability_fn": tf.nn.softmax
                        }

                    Here "probability_fn" can also be set to the string name
                    or module path to a probability function.

                "alignment_history": bool
                    whether to store alignment history from all time steps
                    in the final output state. (Stored as a time major
                    `TensorArray` on which you must call `stack()`.)

                "output_attention": bool
                    If `True` (default), the output at each time step is
                    the attention value. This is the behavior of Luong-style
                    attention mechanisms. If `False`, the output at each
                    time step is the output of `cell`.  This is the
                    beahvior of Bhadanau-style attention mechanisms.
                    In both cases, the `attention` tensor is propagated to
                    the next time step via the state and is used there.
                    This flag only controls whether the attention mechanism
                    is propagated up to the next cell in an RNN stack or to
                    the top RNN output.
        """
        hparams = RNNDecoderBase.default_hparams()
        hparams["name"] = "copynet_rnn_decoder"
        hparams["copying"] = {
            "copying_probability_history": False,
            "coverity_state": True,
            "selective_read": True,
        }
        return hparams

    # pylint: disable=arguments-differ
    def _get_beam_search_cell(self, beam_width):
        """Returns the RNN cell for beam search decoding.
        """
        with tf.variable_scope(self.variable_scope, reuse=True):
            copying_mechanism = self._cell.copying_mechanism
            if self._cell._is_multi:
                bs_copying_mechanism = [
                    c._get_beam_search_copying_mechanism(beam_width)
                    for c in copying_mechanism]
            else:
                bs_copying_mechanism = \
                    copying_mechanism._get_beam_search_copying_mechanism(
                        beam_width)

            bs_copy_cell = CopyNetWrapper(
                self._cell._cell,
                bs_copying_mechanism,
                self._vocab_size,
                initial_cell_state=self._initial_cell_state,
                generating_layer=self._cell.generating_layer,
                **self._copy_cell_kwargs)

            self._beam_search_cell = bs_copy_cell

            return bs_copy_cell

    def _wrap_selective_read_contexts(inputs, copying_probability, last_ids):
        if self._cell._is_multi:
            all_copying_probabilities = copying_probability
        else:
            all_copying_probabilities = [copying_probability]

        selective_read_contexts = []
        for copying_mechanism, copying_probability in zip(
                self._cell._copying_mechanisms, all_copying_probabilities):
            selective_read_contexts.append(
                copying_mechanism.selective_read(
                    last_ids, copying_probability))

        if self._selective_read_input_fn is None:
            inputs = tf.concat(
                [inputs] + selective_read_contexts, -1)
        else:
            inputs = self._selective_read_input_fn(
                inputs,
                selective_read_contexts if self._cell._is_multi else
                selective_read_contexts[0])

        return inputs

    def initialize(self, name=None):
        initial_finished, initial_inputs = self._helper.initialize()

        flat_initial_state = nest.flatten(self._initial_state)
        dtype = flat_initial_state[0].dtype
        batch_size = tf.shape(flat_initial_state[0])[0]
        initial_state = self._cell.zero_state(
            batch_size=batch_size, dtype=dtype)
        initial_state = initial_state.clone(cell_state=self._initial_state)

        if self._selective_read:
            last_ids = tf.fill([batch_size], -1)
            initial_inputs = self._wrap_selective_read_contexts(
                initial_inputs, initial_state.copying_probability, last_ids)

        return initial_finished, initial_inputs, initial_state

    def step(self, time, inputs, state, name=None):
        wrapper_outputs, wrapper_state = self._cell(inputs, state)
        logits = self._output_layer(wrapper_outputs)
        sample_ids = self._helper.sample(
            time=time, outputs=logits, state=wrapper_state)
        reach_max_time = tf.equal(time+1, self.max_decoding_length)

        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=logits,
            state=wrapper_state,
            sample_ids=sample_ids,
            reach_max_time=reach_max_time)

        if self._selective_read:
            next_inputs = self._wrap_selective_read_contexts(
                next_inputs, wrapper_state.copying_probability, sample_ids)

        outputs = CopyNetRNNDecoderOutput(
            logits, sample_ids, wrapper_outputs,
            wrapper_state.copying_probability)

        return (outputs, next_state, next_inputs, finished)

    def finalize(self, outputs, final_state, sequence_lengths):
        return outputs, final_state

    def _memory_size(self):
        memory_size = [cm.memory_size for cm in self._cell._copying_mechanisms]
        return self._cell._item_or_tuple(memory_size)

    @property
    def output_size(self):
        return CopyNetRNNDecoderOutput(
            logits=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape,
            cell_output=self._cell.output_size,
            copying_probability=self._memory_size())

    @property
    def output_dtype(self):
        """Types of output of one step.
        """
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return CopyNetRNNDecoderOutput(
            logits=nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            sample_id=self._helper.sample_ids_dtype,
            cell_output=nest.map_structure(
                lambda _: dtype, self._cell.output_size),
            copying_probability=nest.map_structure(
                lambda _: dtype, self._memory_size()))

    def zero_state(self, batch_size, dtype):
        """Returns zero state of the basic cell.
        Equivalent to :attr:`decoder.cell._cell.zero_state`.
        """
        return self._cell._cell.zero_state(batch_size=batch_size, dtype=dtype)

    def wrapper_zero_state(self, batch_size, dtype):
        """Returns zero state of the attention-wrapped cell.
        Equivalent to :attr:`decoder.cell.zero_state`.
        """
        return self._cell.zero_state(batch_size=batch_size, dtype=dtype)

    @property
    def state_size(self):
        """The state size of the basic cell.
        Equivalent to :attr:`decoder.cell._cell.state_size`.
        """
        return self._cell._cell.state_size

    @property
    def wrapper_state_size(self):
        """The state size of the attention-wrapped cell.
        Equivalent to :attr:`decoder.cell.state_size`.
        """
        return self._cell.state_size
