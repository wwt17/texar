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
CopyNets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=not-context-manager, protected-access, invalid-name

import collections

import numpy as np

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.contrib.seq2seq import tile_batch

__all__ = [
    "CopyingMechanism",
    "BasicCopyingMechanism",
    "CopyNetWrapperState",
    "CopyNetWrapper",
]


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
    """Convert to tensor and possibly mask `memory`.
    This function is copied from Tensorflow attention_wrapper.py.

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
    memory = nest.map_structure(
        lambda m: tf.convert_to_tensor(m, name="memory"), memory)
    if memory_sequence_length is not None:
        memory_sequence_length = tf.convert_to_tensor(
            memory_sequence_length, name="memory_sequence_length")
    if check_inner_dims_defined:
        def _check_dims(m):
            if not m.get_shape()[2:].is_fully_defined():
                raise ValueError(
                    "Expected memory %s to have fully defined inner dims, "
                    "but saw shape: %s" % (m.name, m.get_shape()))
        nest.map_structure(_check_dims, memory)
    if memory_sequence_length is None:
        seq_len_mask = None
    else:
        seq_len_mask = tf.sequence_mask(
            memory_sequence_length,
            maxlen=tf.shape(nest.flatten(memory)[0])[1],
            dtype=nest.flatten(memory)[0].dtype)
        seq_len_batch_size = (
            memory_sequence_length.shape[0].value
            or tf.shape(memory_sequence_length)[0])
    def _maybe_mask(m, seq_len_mask):
        rank = m.get_shape().ndims
        rank = rank if rank is not None else tf.rank(m)
        extra_ones = tf.ones(rank - 2, dtype=tf.int32)
        m_batch_size = m.shape[0].value or tf.shape(m)[0]
        if memory_sequence_length is not None:
            message = ("memory_sequence_length and memory tensor batch sizes "
                       "do not match.")
            with tf.control_dependencies([
                    tf.assert_equal(
                        seq_len_batch_size, m_batch_size, message=message)]):
                seq_len_mask = tf.reshape(
                    seq_len_mask,
                    tf.concat((tf.shape(seq_len_mask), extra_ones), 0))
                return m * seq_len_mask
        else:
            return m
    return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
    """This function is copied from Tensorflow attention_wrapper.py.
    """
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
    """An abstract base CopyingMechanism class providing common functionality.
    This class is modified from _BaseAttentionMechanism in Tensorflow
    attention_wrapper.py.
    Common functionality includes:
        1. Storing the query and memory layers.
        2. Preprocessing and storing the memory_state.

    Args:
        memory_id: The memory ids to copy from.  This tensor should be
            shaped `[batch_size, max_time]`.
        memory_state: The memory states to get copying information; usually
            the output of an encoder.  This tensor should be shaped
            `[batch_size, max_time, state_dim]`.
        memory_sequence_length (optional): Sequence lengths for the batch
            entries in memory.  If provided, the memory_state tensor rows
            are masked with zeros for values past the respective sequence
            lengths.
        memory_layer (optional): Instance of
            :tf_main:`tf.layers.Layer <layers/Layer>` (may be None). The
            layer's depth must match the depth of `query_layer`. If
            `memory_layer` is not provided, the shape of `memory_state` must
            match that of `query_layer`.
        query_layer (optional): Callable.  Instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`.
            The layer's depth must match the depth of `memory_layer`.
            If `query_layer` is not provided, the shape of `query` must
            match that of `memory_layer`.
        check_inner_dims_defined (bool, optional): Python boolean.
            If `True`, the `memory_state` argument's shape is checked to
            ensure all but the two outermost dimensions are fully defined.
        scope (optional): Instance of
            :tf_main:`tf.VariableScope <VariableScope>` (may be None). The
            scope to create variables within. If not provided, a
            variable scope with name `name` is used.
        name (str, optional): Name to use when creating ops.
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
        self._memory_layer = memory_layer
        self._query_layer = query_layer
        self.dtype = memory_layer.dtype

        if scope is None:
            with tf.variable_scope(name, "basic_copying_mechanism") as scope:
                pass
        self._variable_scope = scope
        self._name = name

        with tf.name_scope(
                name, "CopyingMechanismInit",
                nest.flatten(memory_id) + nest.flatten(memory_state)):
            with tf.variable_scope(self._variable_scope, reuse=tf.AUTO_REUSE):
                self._memory_state = _prepare_memory(
                    memory_state, memory_sequence_length,
                    check_inner_dims_defined=check_inner_dims_defined)
                self._memory_id = memory_id
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

    def __call__(self, query, coverage_state):
        """Obtain copying scores based on the query and coverage state.

        Args:
            query: Tensor of dtype matching `self.keys` and shape
                `[batch_size, query_depth]`, where `query_depth` equals
                `num_units`. The query vectors.
            coverage_state: Tensor of dtype matching `self.keys` and shape
                `[batch_size, memory_size, coverage_state_dim]`, where
                `memory_size` is memory's `max_time` and `coverage_state_dim`
                is the depth of coverage states. The previous coverage states.
                If `()` is passed, coverage state is disabled.

        Returns:
            Copying scores for each memory location. Tensor of dtype matching
            `self.keys` and shape `[batch_size, memory_size]`, where
            `memory_size` is memory's `max_time`.
        """
        raise NotImplementedError

    def map_to_vocab(self, copying_probability, vocab_size):
        """Mapping copying probability distribution over memory to the
        probability distribution over generating vocabulary.

        Args:
            copying_probability: Tensor of shape `[batch_size, memory_size]`.
            vocab_size (int): The size of generating vocabulary.

        Returns:
            The result probability over generating vocabulary.
        """
        raise NotImplementedError

    def initial_copying_probability(self, batch_size, dtype):
        """Creates the initial copying probability.
        
        Args:
            batch_size: int scalar, the batch_size.
            dtype: The dtype.

        Returns:
            A `dtype` tensor shaped `[batch_size, memory_size]`
            (`memory_size` is the memory's `max_time`).
        """
        raise NotImplementedError

    def initial_coverage_state(self, batch_size, dtype):
        """Creates the initial coverage state.
 
        Args:
            batch_size: int scalar, the batch_size.
            dtype: The dtype.

        Returns:
            A `dtype` tensor shaped `[batch_size, coverage_state_size]`.
        """
        raise NotImplementedError

    def update_coverage_state(
            self, coverage_state, copying_probability, cell_outputs):
        """Update the coverage state. The coverage_cell is used here.

        Args:
            coverage_state: Tensor of shape
                `[batch_size, memory_size * coverage_state_dim]`. The previous
                coverage state to update.
            copying_probability: Tensor of shape `[batch_size, memory_size]`.
                The copying probability this step.
            state: Tensor of shape `[batch_size, dim]`, where `dim` is the
                depth of the state. The cell output state used.

        Returns:
            The updated coverage state, of the same shape as `coverage_state`.
        """
        raise NotImplementedError

    def selective_read(self, last_ids, copying_probability):
        """This method implements Eq.(9) in the paper.

        Args:
            last_ids: int Tensor of shape `[batch_size]`. The ids obtained in
                previous step, i.e. y_{t-1}.
            copying_probability: Tensor of shape `[batch_size, memory_size]`.
                The copying probability of previous step.

        Returns:
            Tensor of shape `[batch_size, memory_state_dim]`, where the
            `memory_state_dim` is the depth of `memory_state`.
            The selective read context.
        """
        raise NotImplementedError

    def _get_beam_search_copying_mechanism(self):
        """Get the tiled instance of this instance used in beam searching.
        All parameters are shared with the original one.

        Args:
            beam_width (int): The beam width. Used as the multiple of tiling.

        Returns:
            A tiled instance.
        """
        raise NotImplementedError

    @property
    def memory_layer(self):
        """The memory layer.
        """
        return self._memory_layer

    @property
    def query_layer(self):
        """The query layer.
        """
        return self._query_layer

    @property
    def memory_id(self):
        """The memory ids.
        """
        return self._memory_id

    @property
    def memory_state(self):
        """The memory states.
        """
        return self._memory_state

    @property
    def keys(self):
        """The (transformed) memory state used to calculate copying scores.
        """
        return self._keys

    @property
    def batch_size(self):
        """The batch size.
        """
        return self._batch_size

    @property
    def memory_size(self):
        """The memory size, i.e. the max_time of memory.
        """
        return self._memory_size

    @property
    def coverage_state_size(self):
        """The state size of coverage states.
        """
        raise NotImplementedError

    @property
    def variable_scope(self):
        """The variable scope used to create variables within.
        """
        return self._variable_scope


class BasicCopyingMechanism(CopyingMechanism):
    """Basic copying mechanism.
    Implements the copying mechanism proposed in
    https://arxiv.org/pdf/1603.06393.pdf. However, there're some differences:
        1. "selective read" is not implemented.
        2. Additional (optional) coverage module is implemented.

    Args:
        num_units: The depth of copying mechanism. This is used as the depth
            of transformed memory_state, therefore must match expected the
            query depth.
        memory_id: The memory ids to copy from.  This tensor should be
            shaped `[batch_size, max_time]`.
        memory_state: The memory states to get copying information; usually
            the output of an encoder.  This tensor should be shaped
            `[batch_size, max_time, state_dim]`.
        memory_sequence_length (optional): Sequence lengths for the batch
            entries in memory.  If provided, the memory_state tensor rows
            are masked with zeros for values past the respective sequence
            lengths.
        activation (optional): A callable. The activation function used after
            merge transformed copying states with coverage states. Default to
            :tf_main:`tf.tanh <tanh>` as described in the paper.
        coverage_state_dim (int, optional): The depth of coverage state.
            Ignored if coverage state is disabled.
        coverage_cell (RNNCell, optional): An instance of
            :tf_main:`RNNCell <nn/rnn_cell/RNNCell>`. The cell used to update
            coverage states. Ignored if coverage state is disabled.
        score_mask_value (optional): The float scalar used to mask output
            copying scores in positions after `memory_sequence_length`. If not
            provided, -inf is used.
        scope (optional): Instance of
            :tf_main:`tf.VariableScope <VariableScope>` (may be None). The
            scope to create variables within. If not provided, a
            variable scope with name `name` is used.
        name (str, optional): Name to use when creating ops.
    """

    def __init__(self,
                 num_units,
                 memory_id,
                 memory_state,
                 memory_sequence_length=None,
                 activation=tf.tanh,
                 coverage_state_dim=None,
                 coverage_cell=None,
                 score_mask_value=None,
                 scope=None,
                 name="BasicCopyingMechanism"):
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
        self._coverage_state_dim = coverage_state_dim
        self._coverage_cell = coverage_cell

        if score_mask_value is None:
            score_mask_value = -np.inf
        self._score_mask_value = score_mask_value

    def __call__(self, query, coverage_state):
        """Obtain copying scores (i.e. phi_c in the paper) based on the query
        and coverage state.

        Args:
            query: Tensor of dtype matching `self.keys` and shape
                `[batch_size, query_depth]`, where `query_depth` equals
                `num_units`. The query vectors.
            coverage_state: Tensor of dtype matching `self.keys` and shape
                `[batch_size, memory_size, coverage_state_dim]`, where
                `memory_size` is memory's `max_time` and `coverage_state_dim`
                is the depth of coverage states. The previous coverage states.
                If `()` is passed, coverage state is disabled.

        Returns:
            Copying scores for each memory location. Tensor of dtype matching
            `self.keys` and shape `[batch_size, memory_size]`, where
            `memory_size` is memory's `max_time`.
        """
        with tf.variable_scope(
                self._variable_scope,
                values=[query, coverage_state],
                reuse=tf.AUTO_REUSE):
            keys = self._keys
            if coverage_state is not ():
                coverage_state = tf.reshape(
                    coverage_state,
                    [tf.shape(coverage_state)[0], -1,
                        self._coverage_state_dim])
                keys = keys + tf.layers.dense(
                    coverage_state,
                    units=self._num_units,
                    activation=None,
                    use_bias=False)
            keys = self._activation(keys)
            score = tf.einsum("bim,bm->bi", self._memory_state, query)
            score = _maybe_mask_score(
                score, self._memory_sequence_length, self._score_mask_value)
            return score

    def map_to_vocab(self, copying_probability, vocab_size):
        """Mapping copying probability distribution over memory to the
        probability distribution over generating vocabulary. This method
        assumes the ordered vocabulary of memory_id is a prefix of the
        generating vocabulary. Therefore
        :tf_main:`tf.scatter_nd <scatter_nd>` is used.

        Args:
            copying_probability: Tensor of shape `[batch_size, memory_size]`.
            vocab_size (int): The size of generating vocabulary.

        Returns:
            The result probability over generating vocabulary.
        """
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
        """Creates the initial copying probability.
 
        Args:
            batch_size: int scalar, the batch_size.
            dtype: The dtype.

        Returns:
            A `dtype` tensor shaped `[batch_size, memory_size]`
            (`memory_size` is the memory's `max_time`).
        """
        return tf.zeros([batch_size, self._memory_size], dtype=dtype,
                        name="initial_copying_probability")

    def initial_coverage_state(self, batch_size, dtype):
        """Creates the initial coverage state.
 
        Args:
            batch_size: int scalar, the batch_size.
            dtype: The dtype.

        Returns:
            A `dtype` tensor shaped `[batch_size, coverage_state_size]`.
        """
        return tf.zeros(
            [batch_size, self._memory_size * self._coverage_state_dim],
            dtype=dtype, name="initial_coverage_state")

    def update_coverage_state(self, coverage_state, copying_probability, state):
        """Update the coverage state. The coverage_cell is used here.

        Args:
            coverage_state: Tensor of shape
                `[batch_size, memory_size * coverage_state_dim]`. The previous
                coverage state to update.
            copying_probability: Tensor of shape `[batch_size, memory_size]`.
                The copying probability this step.
            state: Tensor of shape `[batch_size, dim]`, where `dim` is the
                depth of the state. The cell output state used.

        Returns:
            The updated coverage state, of the same shape as `coverage_state`.
        """
        coverage_state_shape = tf.shape(coverage_state)
        shape = tf.shape(copying_probability)
        state = tf.broadcast_to(
            tf.expand_dims(state, 1),
            tf.concat([shape, [state.shape[-1]]], -1))
        coverage_state = tf.reshape(
            coverage_state, [-1, self._coverage_state_dim])
        copying_probability = tf.reshape(copying_probability, [-1, 1])
        state = tf.reshape(state, [-1, state.shape[-1]])
        _, coverage_state = self._coverage_cell(
            tf.concat([copying_probability, state], -1), coverage_state)
        coverage_state = tf.reshape(coverage_state, coverage_state_shape)
        return coverage_state

    def selective_read(self, last_ids, copying_probability):
        """This method implements Eq.(9) in the paper.

        Args:
            last_ids: int Tensor of shape `[batch_size]`. The ids obtained in
                previous step, i.e. y_{t-1}.
            copying_probability: Tensor of shape `[batch_size, memory_size]`.
                The copying probability of previous step.

        Returns:
            Tensor of shape `[batch_size, memory_state_dim]`, where the
            `memory_state_dim` is the depth of `memory_state`.
            The selective read context.
        """
        dtype = copying_probability.dtype
        int_mask = tf.cast(
            tf.equal(tf.expand_dims(last_ids, 1),
                     tf.cast(self._memory_id, last_ids.dtype)),
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
        """Get the tiled instance of this instance used in beam searching.
        All parameters are shared with the original one.

        Args:
            beam_width (int): The beam width. Used as the multiple of tiling.

        Returns:
            A tiled instance.
        """
        return BasicCopyingMechanism(
            self._num_units,
            tile_batch(self._memory_id, beam_width),
            tile_batch(self._memory_state, beam_width),
            memory_sequence_length=
                None if self._memory_sequence_length is None else
                tile_batch(self._memory_sequence_length, beam_width),
            activation=self._activation,
            coverage_state_dim=self._coverage_state_dim,
            coverage_cell=self._coverage_cell,
            scope=self._variable_scope,
            name="BeamSearch{}".format(self._name))

    @property
    def coverage_state_size(self):
        """`memory_size * coverage_state_dim`.
            (`memory_size` is the memory's `max_time`, and `coverage_state_dim`
             is the depth of coverage state).
            The state size of coverage states.
        """
        return self._memory_size * self._coverage_state_dim


class CopyNetWrapperState(
    collections.namedtuple("CopyNetWrapperState",
                           ("cell_state", "time", "copying_probability",
                            "copying_probability_history", "coverage_state"))):
    """`namedtuple` storing the state of a
    :class:`~texar.modules.CopyNetWrapper`.

    Attributes:
        cell_state: The state of the wrapped
            :tf_main:`RNNCell <nn/rnn_cell/RNNCell>` at the previous time step.
        time: int32 scalar containing the current time step.
        copying_probability: A single or tuple of `Tensor` (s) containing the
            copying probability obtained at the previous time step for each
            copying mechanism.
        copying_probability_history: (if enabled) a single or tuple of
            `TensorArray` (s) containing copying probability matrices from all
            time steps for each copying mechanism. Call `stack()` on each to
            convert to a `Tensor`.
        coverage_state: A single or tuple of nested objects
            containing coverage state for each copying mechanism.
            The objects may contain Tensors or `TensorArray` s.
    """

    def clone(self, **kwargs):
        """Clone this object, overriding components provided by `kwargs`.
        The new state fields' shape must match original state fields' shape.
        This will be validated, and original fields' shape will be propagated to
        new fields.
        Example:

        .. code-block:: python

            initial_state = copynet_wrapper.zero_state(dtype=..., batch_size=...)
            initial_state = initial_state.clone(cell_state=encoder_state)

        Args:
            **kwargs: Any properties of the state object to replace in the
                returned :class:`~texar.modules.CopyNetWrapperState`.
        Returns:
            A new :class:`~texar.modules.CopyNetWrapperState` whose properties
            are the same as this one, except any overridden properties as
            provided in `kwargs`.
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
    """Wraps another :tf_main:`RNNCell <nn/rnn_cell/RNNCell>` with CopyNet.
    This wrapper implements the main process of CopyNet model described in
    https://arxiv.org/pdf/1608.05859.pdf. To completely implement the CopyNet
    model, use this wrapper to wrap an
    :tf_main:`AttentionWrapper <contrib/seq2seq/AttentionWrapper>` cell which
    attends on the memory, and a
    :class:`~texar.modules.BasicCopyingMechanism` with the same memory. Note
    that such a model is not exactly the same as the model described in the
    paper. To see their differences, please refer to
    :class:`~texar.modules.BasicCopyingMechanism`. Other instances of
    :class:`~texar.modules.CopyingMechanism` are also supported.

    This function is copied and modified from Tensorflow
    :tf_main:`AttentionWrapper <contrib/seq2seq/AttentionWrapper>` in
    attention_wrapper.py.

    Args:
        cell: An instance of :tf_main:`RNNCell <nn/rnn_cell/RNNCell>`.
        copying_mechanism: A list of :class:`~texar.modules.CopyingMechanism`
            or a single instance.
        vocab_size (int): The size of generating vocabulary.
        copying_probability_history (bool, optional): Python boolean, whether
            to store copying probability history from all time steps in the
            final output state (currently stored as a time major
            `TensorArray` on which you must call `stack()`).
        coverage (bool, optional): Python boolean, whether to add coverage
            modules. Default to `False`.
        initial_cell_state (optional): The initial state value to use for the
            cell when the user calls :meth:`zero_state`. Note that if this value
            is provided now, and the user uses a `batch_size` argument of
            :meth:`zero_state` which does not match the batch size of
            `initial_cell_state`, proper behavior is not guaranteed.
        generating_layer (optional): A callable taking the cell output as input
            to generate the generating scores at each time step. If `None`
            (default), a dense layer is used.
        output_layer (optional): A callable taking the output probability as
            input to generate the output at each time step. If `None`
            (default), the output probability is the output.
        name (str, optional): Name to use when creating ops.
    """

    def __init__(self,
                 cell,
                 copying_mechanism,
                 vocab_size,
                 copying_probability_history=False,
                 coverage=False,
                 initial_cell_state=None,
                 generating_layer=None,
                 output_layer=None,
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
        self._output_layer = output_layer

        self._cell = cell
        self._copying_mechanism = copying_mechanism
        self._copying_mechanisms = copying_mechanisms
        self._copying_probability_history = copying_probability_history
        self._coverage = coverage

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
                    "the BeamSearchDecoder?  You may need to tile your "
                    "initial state via the tf.contrib.seq2seq.tile_batch "
                    "function with argument multiple=beam_width.")
                with tf.control_dependencies(
                        self._batch_size_checks(state_batch_size,
                                                error_message)):
                    self._initial_cell_state = nest.map_structure(
                        lambda s:
                            tf.identity(s, name="check_initial_cell_state"),
                        initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [tf.assert_equal(batch_size,
                                copying_mechanism.batch_size,
                                message=error_message)
                for copying_mechanism in self._copying_mechanisms]

    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the
        :class:`~texar.modules.CopyingMechanism` (s)
        were passed to the constructor.

        Args:
            seq: A non-empty sequence of items or generator.

        Returns:
            Either the values in the sequence as a tuple if
            :class:`~texar.modules.CopyingMechanism` (s)
            were passed to the constructor as a sequence or the singular
            element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]

    @property
    def copying_mechanism(self):
        """The copying mechanism(s) passed into the constructor.
        """
        return self._copying_mechanism

    @property
    def generating_layer(self):
        """The generating layer.
        """
        return self._generating_layer

    @property
    def output_layer(self):
        """The output layer.
        """
        return self._output_layer

    @property
    def vocab_size(self):
        """The generating vocabulary size.
        """
        return self._vocab_size

    @property
    def output_size(self):
        """The output size, which is equal to the generating vocabulary size.
        """
        return self._vocab_size

    @property
    def state_size(self):
        """The state size.

        Returns:
            An :class:`~texar.modules.CopyNetWrapperState` tuple containing
            shapes used by this object.
        """
        return CopyNetWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            copying_probability=self._item_or_tuple(
                c.memory_size for c in self._copying_mechanisms),
            copying_probability_history=self._item_or_tuple(
                c.memory_size if self._copying_probability_history else ()
                for c in self._copying_mechanisms),  # sometimes a TensorArray
            coverage_state=self._item_or_tuple(
                c.coverage_state_size if self._coverage else ()
                for c in self._copying_mechanisms))

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this
        :class:`~texar.modules.CopyNetWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an :class:`~texar.modules.CopyNetWrapper`
        with a
        :tf_main:`BeamSearchDecoder <contrib/seq2seq/BeamSearchDecoder>`.

        Args:
            batch_size: `0D` integer tensor: the batch size.
            dtype: The internal state data type.

        Returns:
            An :class:`~texar.modules.CopyNetWrapperState` tuple containing
            zeroed out tensors and, possibly, empty `TensorArray` objects.

        Raises:
            ValueError: (or, possibly at runtime, InvalidArgument), if
                `batch_size` does not match the output size of the encoder
                passed to the wrapper object at initialization time.
        """
        with tf.name_scope(type(self).__name__ + "ZeroState",
                           values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                "When calling zero_state of CopyNetWrapper %s: "
                % self._base_name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the requested batch size.  Are you using "
                "the BeamSearchDecoder?  If so, make sure your encoder output "
                "has been tiled to beam_width via "
                "tf.contrib.seq2seq.tile_batch, and the batch_size= argument "
                "passed to zero_state is batch_size * beam_width.")
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
                coverage_state=self._item_or_tuple(
                    copying_mechanism.initial_coverage_state(batch_size, dtype)
                    if self._coverage else ()
                    for copying_mechanism in self._copying_mechanisms))

    def call(self, inputs, state):
        """Perform a step of CopyNet-wrapped RNN.

            - Step 1: Call the wrapped `cell` with this input and its \
                previous state.
            - Step 2: Get generating scores through `generating_layer`.
            - Step 3: Get copying scores through `attention_mechanism`.
            - Step 4: Calculate the generating and copying probabilities by \
                passing the score through normalization.
            - Step 5 (Optional): Calculate the output by passing the output \
                probability through `output_layer`.

        Args:
            inputs: (Possibly nested tuple of) Tensor, the input at this time
                step.
            state: An instance of :class:`~texar.modules.CopyNetWrapperState`
                containing tensors from the previous time step.

        Returns:
            `(output, next_state)`, where

            - `output`: if `output_layer` is provided, it is the result after \
                applying `output_layer` to the output probability; otherwise \
                it is simply the output probability.
            - `next_state`: an instance of \
                :class:`~texar.modules.CopyNetWrapperState` \
                containing the state calculated at this time step.

        Raises:
            TypeError: If `state` is not an instance of
                :class:`~texar.modules.CopyNetWrapperState`.
        """
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of "
                            "CopyNetWrapperState. Received type {} instead."
                            .format(type(state)))

        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or
            tf.shape(cell_output)[0])
        error_message = (
            "When applying CopyNetWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                cell_output, name="checked_cell_output")

        # generating mode
        generating_score = self._generating_layer(cell_output)
        max_generating_score = tf.reduce_max(
            generating_score, axis=1, keepdims=True)
        max_score = max_generating_score

        # copying mode
        if self._is_multi:
            previous_coverage_state = state.coverage_state
            previous_copying_probability_history = \
                state.copying_probability_history
        else:
            previous_coverage_state = [state.coverage_state]
            previous_copying_probability_history = \
                [state.copying_probability_history]

        all_copying_scores = []
        for i, copying_mechanism in enumerate(self._copying_mechanisms):
            copying_score = copying_mechanism(
                cell_output, coverage_state=previous_coverage_state[i])
            max_copying_score = tf.reduce_max(
                copying_score, axis=1, keepdims=True)
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
        all_next_coverage_states = []
        maybe_all_histories = []
        for i, (copying_mechanism, exp_copying_score) in enumerate(zip(
                self._copying_mechanisms, all_exp_copying_scores)):
            copying_probability = exp_copying_score / sum_exp_score
            all_copying_probabilities.append(copying_probability)
            next_coverage_state = copying_mechanism.update_coverage_state(
                    previous_coverage_state[i], copying_probability,
                    cell_output) \
                    if self._coverage else ()
            copying_probability_history = \
                previous_copying_probability_history[i].write(
                    state.time, copying_probability) \
                if self._copying_probability_history else ()

            all_next_coverage_states.append(next_coverage_state)
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
            coverage_state=self._item_or_tuple(all_next_coverage_states))

        output = output_probability
        if self._output_layer is not None:
            output = self._output_layer(output)

        return output, next_state
