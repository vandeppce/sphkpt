# Copyright 2007-2020 The scikit-learn developers.
# Copyright 2020 Google LLC.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""Isotonic optimization routines in Numba."""

import warnings
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
'''
# pylint: disable=g-import-not-at-top
try:
  from numba import njit
except ImportError:
  warnings.warn("Numba could not be imported. Code will run much more slowly."
                " To install, please run 'pip install numba'.")

  # If Numba is not available, we define a dummy 'njit' function.
  def njit(func):
    return func


# Copied from scikit-learn with the following modifications:
# - use decreasing constraints by default,
# - do not return solution in place, rather save in array `sol`,
# - avoid some needless multiplications.


@njit
'''
@tf.function
def isotonic_l2(y, sol):
  """Solves an isotonic regression problem using PAV.

  Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.

  Args:
    y: input to isotonic regression, a 1d-array.
    sol: where to write the solution, an array of the same size as y.
  """
  yy = tf.reshape(y, (16384, 1))
  soll = tf.reshape(sol, (16384, 1))
  # yy = tf.split(yy, num_or_size_splits = 16384, axis = 0)
  # soll = tf.split(soll, num_or_size_splits = 16384, axis = 0)

  n = y.shape[0]
  n = np.int32(128 * 128)
  target = tf.reshape(tf.keras.backend.arange(n), (16384, 1))
  # target = tf.split(target, num_or_size_splits = 16384, axis = 0)
  c = tf.reshape(tf.ones(n), (16384, 1))
  # c = tf.split(c, num_or_size_splits = 16384, axis = 0)
  sums = tf.zeros((16384, 1))
  # sums = tf.split(sums, num_or_size_splits = 16384, axis = 0)
  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.

  for i in range(n):
    tf.tensor_scatter_nd_update(soll, tf.constant([i]), tf.gather(yy, i))
    # sums = soll
    tf.tensor_scatter_nd_update(sums, tf.constant([i]), tf.gather(yy, i))
  
  i = 0
  while tf.less(i, n):
    k = tf.gather(target, i) + 1
    if k == n:
      break
    if tf.less(tf.gather(soll, k), tf.gather(soll, i)):
      i = k
      continue
    sum_y = tf.gather(sums, i)
    sum_c = tf.gather(c, i)
    
    while True:
      # We are within an increasing subsequence.
      prev_y = tf.gather(soll, k)
      sum_y += tf.gather(sums, k)
      sum_c += tf.gather(c, k) 
      k = tf.gather(target, k) + 1
      if k == n or prev_y > tf.gather(soll, k):
        # Non-singleton increasing subsequence is finished,
        # update first entry.
        tf.tensor_scatter_nd_update(soll, [i], sum_y / sum_c)
        tf.tensor_scatter_nd_update(sums, [i], sum_y)
        tf.tensor_scatter_nd_update(c, [i], sum_c)
        tf.tensor_scatter_nd_update(target, [i], k - 1)
        tf.tensor_scatter_nd_update(target, [k - 1], i)
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = tf.gather(target, i - 1)
        # Otherwise, restart from the same point.
        break
      
  # Reconstruct the solution.
  i = 0
  while i < n:
    k = target[i] + 1
    sol[i + 1 : k] = sol[i]
    i = k


# @njit
def _log_add_exp(x, y):
  """Numerically stable log-add-exp."""
  larger = max(x, y)
  smaller = min(x, y)
  return larger + np.log1p(np.exp(smaller - larger))


# Modified implementation for the KL geometry case.
# @njit
def isotonic_kl(y, w, sol):
  """Solves isotonic optimization with KL divergence using PAV.

  Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.

  Args:
    y: input to isotonic optimization, a 1d-array.
    w: input to isotonic optimization, a 1d-array.
    sol: where to write the solution, an array of the same size as y.
  """
  n = y.shape[0]
  target = np.arange(n)
  lse_y_ = np.zeros(n)
  lse_w_ = np.zeros(n)

  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.

  for i in range(n):
    sol[i] = y[i] - w[i]
    lse_y_[i] = y[i]
    lse_w_[i] = w[i]

  i = 0
  while i < n:
    k = target[i] + 1
    if k == n:
      break
    if sol[i] > sol[k]:
      i = k
      continue
    lse_y = lse_y_[i]
    lse_w = lse_w_[i]
    while True:
      # We are within an increasing subsequence.
      prev_y = sol[k]
      lse_y = _log_add_exp(lse_y, lse_y_[k])
      lse_w = _log_add_exp(lse_w, lse_w_[k])
      k = target[k] + 1
      if k == n or prev_y > sol[k]:
        # Non-singleton increasing subsequence is finished,
        # update first entry.
        sol[i] = lse_y - lse_w
        lse_y_[i] = lse_y
        lse_w_[i] = lse_w
        target[i] = k - 1
        target[k - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    k = target[i] + 1
    sol[i + 1 : k] = sol[i]
    i = k
