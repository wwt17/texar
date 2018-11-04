#!/usr/bin/env python3
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
"""Triggers.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle

try:
    import queue
except ImportError:
    import Queue as queue

#pylint: disable=invalid-name, too-many-arguments, too-many-locals

__all__ = [
    "Trigger",
    "BestEverConvergenceTrigger",
]


class Trigger(object):
    """This is the base class of all triggers. A trigger maintains some
    user-defined :attr:`user_state` and does some :attr:`action` when certain
    condition is met. Specifically, the user calls the trigger periodically.
    Every time the trigger is called, it will send all arguments to
    :meth:`_predicate`, which returns a boolean value indicates whether the
    condition is met. Once the condition is met, the trigger will then execute
    `user_state = action(user_state)` to update the :attr:`user_state`.
    :attr:`user_state` should completely define the current state of the
    trigger, and, therefore, enables saving and restoring :attr:`user_state`.
    It is the user's responsibility to keep :attr:`action` away from any
    possible corruption of restored state.

    Args:
        initial_user_state: A (any kind of picklable) object representing the
            initial :attr:`user_state`.
        action (function): A function which is called to update
            :attr:`user_state` every time the trigger is triggered. See above
            for detailed explanation.
    .. document private functions
    .. automethod:: __call__
    """

    def __init__(self, initial_user_state, action):
        self._user_state = initial_user_state
        self._action = action

    def _predicate(self, *args, **kwargs):
        """Returns True when the condition is met and we should do something.
        """
        raise NotImplementedError

    def trigger(self):
        """Executes `user_state = action(user_state)`. User can manually call
        this method to trigger it.
        """
        self._user_state = self._action(self._user_state)

    def __call__(self, *args, **kwargs):
        """The trigger must be called to update the internal state and
        automatically triggers when the condition is found met.
        """
        pred = self._predicate(*args, **kwargs)
        if pred:
            self.trigger()
        return pred

    def _make_state(self, names):
        return {name: getattr(self, name) for name in names}

    @property
    def _state_names(self):
        """Returns a list of names of attributes of the trigger object that can
        be saved and restored as trigger state.
        """
        return ['_user_state']

    @property
    def state(self):
        """The current state which can be used to save and restore the trigger.
        The state is consisted of the internal state used to determine whether
        the condition is met, and the user-defined :attr:`user_state`.
        """
        return self._make_state(self._state_names)

    @property
    def user_state(self):
        """The user-defined :attr:`user_state`.
        """
        return self._user_state

    def restore_from_state(self, state):
        """Restore the trigger state from the previous saved state.

        Args:
            state: The state previously obtained by :attr:`state`.
        """
        for name, value in state.items():
            setattr(self, name, value)

    def save_to_pickle(self, file):
        """Write a pickled representation of the state of the trigger to the
        open file-like object :attr:`file`.

        Args:
            file: The open file-like object to which we write. As described in
                pickle official document, it must have a `write()` method that
                accepts a single string argument.
        """
        pickle.dump(self.state, file)

    def restore_from_pickle(self, file):
        """Read a string from the open file-like object :attr:`file` and
        restore the trigger state from it.

        Args:
            file: The open file-like object from which we read. As described in
                pickle official document, it must have a `read()` method that
                takes an integer argument, and a `readline()` method that
                requires no arguments, and both methods should return a string.
        """
        self.restore_from_state(pickle.load(file))


class ScheduledStepsTrigger(Trigger):
    """A trigger that triggers at designated steps.
    """
    
    def __init__(self, initial_user_state, action, steps):
        """steps should be a list or tuple in increasing order.
        """
        super(ScheduledTrigger, self).__init__(initial_user_state, action)
        self._steps = steps
        self._advance_steps()

    def _advance_steps(self):
        self._next_step = next(step, None)

    def _predicate(self, step):
        while self._next_step is not None and step < self._next_step:
            self._advance_steps()
        if self._next_step is not None and step == self._next_step:
            return True
        return False


class BestEverConvergenceTrigger(Trigger):
    """A trigger that maintains the best value of a metric. It triggers when
    the best value of the metric has not been updated for at least
    :attr:`threshold_steps`. In order to avoid it triggers two frequently, it
    will not trigger again within :attr:`minimum_interval_steps` once it
    triggers.

    Args:
        initial_user_state: A (any kind of picklable) object representing the
            initial :attr:`user_state`.
        action (function): A function which is called to update
            :attr:`user_state` every time the trigger is triggered.
        threshold_steps (int): Number of steps it should trigger after the best
            value was last updated.
        minimum_interval_steps (int): Minimum number of steps between twice
            firing of the trigger.
        default (optional): The value returned after :attr:`action` exhausted.
            If not provided, the trigger will do nothing when `StopIteration`
            occurs.
    .. document private functions
    .. automethod:: __call__
    """

    def __init__(self, initial_user_state, action, threshold_steps,
                 minimum_interval_steps):
        super(BestEverConvergenceTrigger, self).__init__(
            initial_user_state, action)
        self._threshold_steps = threshold_steps
        self._minimum_interval_steps = minimum_interval_steps
        self._last_triggered_step = None
        self._best_ever_step = None
        self._best_ever_score = None

    def _predicate(self, step, score):
        if self._best_ever_score is None or self._best_ever_score < score:
            self._best_ever_score = score
            self._best_ever_step = step

        if (self._last_triggered_step is None or
                step - self._last_triggered_step >=
                self._minimum_interval_steps) and \
                step - self._best_ever_step >= self._threshold_steps:
            self._last_triggered_step = step
            return True
        return False

    def __call__(self, step, score):
        """The trigger must be called to update the current training step
        (:attr:`step`) and the current value of the maintained metric
        (:attr:`score`).

        Args:
            step (int): Current training step to update. The training step must
                be updated in ascending order.
            score (float): Current value of the maintained metric.

        Returns:
            A tuple `(triggered, retval)`, where boolean `triggered` denotes
            whether triggered this time and `retval` is the return value of the
            action performed this time.
        """
        return super(BestEverConvergenceTrigger, self).__call__(step, score)

    @property
    def _state_names(self):
        return super(BestEverConvergenceTrigger, self)._state_names + [
            '_last_triggered_step', '_best_ever_step', '_best_ever_score']

    @property
    def last_triggered_step(self):
        """The step at which the Trigger last triggered.
        """
        return self._last_triggered_step

    @property
    def best_ever_step(self):
        """The step at which the best-ever score is reached.
        """
        return self._best_ever_step

    @property
    def best_ever_score(self):
        """The best-ever score.
        """
        return self._best_ever_score


class MovingAverageConvergenceTrigger(Trigger):

    def __init__(self, initial_user_state, action, n, threshold,
                 minimum_interval_steps):
        super(MovingAverageConvergenceTrigger, self).__init__(
            initial_user_state, action)
        self._n = n
        self._threshold = threshold
        self._minimum_interval_steps = minimum_interval_steps
        self._last_triggered_step = None
        self._head_queue = queue.Queue(self._n)
        self._head_sum = 0
        self._rear_queue = queue.Queue(self._n)
        self._rear_sum = 0

    def _predicate(self, step, score):
        if self._head_queue.full():
            e = self._head_queue.get()
            self._head_sum -= e
            if self._rear_queue.full():
                self._rear_sum -= self._rear_queue.get()
            self._rear_queue.put(e)
            self._rear_sum += e
        self._head_queue.put(score)
        self._head_sum += score

        if (self._last_triggered_step is None or
                step - self._last_triggered_step
                >= self._minimum_interval_steps) and \
                self._head_queue.full() and self._rear_queue.full() and \
                self._head_sum - self._rear_sum <= self._n * self._threshold:
            self._last_triggered_step = step
            return True
        return False

    @property
    def _state_names(self):
        return super(BestEverConvergenceTrigger, self)._state_names + [
            '_last_triggered_step', '_head_queue', '_head_sum', '_rear_queue',
            '_rear_sum']
