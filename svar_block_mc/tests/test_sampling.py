"""
Tests for `sampling.py`

"""

import jax.numpy as np
from jax import random
from ..sampling import gen_samples_A_L, gen_samples_A_ii_0, gen_samples_C_i
import pytest
import copy


class TestSampling:
    def setup(self):
        self.key = random.PRNGKey(0)
        self.key, *self.keys_A_ii_0 = random.split(self.key, 3)
        self.key, *self.keys_C_i = random.split(self.key, 3)
        self.samples_nb = 100
        self.T = 100
        self.S_C_hat = [np.array([[1.]]), np.array([[2.]])]
        self.C_hat = [np.array([[1.], [1.]]), np.array([[2.], [2.]])]
        self.X_T_X_inv = [np.eye(2), np.eye(2)]

        self.inputs = [self.keys_A_ii_0, self.keys_C_i, self.samples_nb,
                       self.T, self.S_C_hat, self.C_hat, self.X_T_X_inv]

        self.A_L = gen_samples_A_L(*self.inputs)

    def test_same_key(self):
        A_L_new = gen_samples_A_L(*self.inputs)

        for i in range(len(A_L_new)):
            assert (A_L_new[i] == self.A_L[i]).all()

    def test_different_key(self):
        key, *keys_A_ii_0_new = random.split(self.key, 3)
        key, *keys_C_i_new = random.split(key, 3)

        A_L_new = gen_samples_A_L(keys_A_ii_0_new, keys_C_i_new,
                                  self.samples_nb, self.T, self.S_C_hat,
                                  self.C_hat, self.X_T_X_inv)
        for i in range(len(A_L_new)):
            assert (A_L_new[i] != self.A_L[i]).any()

    def test_invalid_lengths(self):
        invalid_input_inds = [0, 1, 5, 6]

        for i in invalid_input_inds:
            inputs_new = copy.copy(self.inputs)
            inputs_new[i] = [np.array([[1.]])]
            with pytest.raises(ValueError):
                gen_samples_A_L(*inputs_new)

    def test_invalid_dims(self):
        invalid_input_inds = [4, 5, 6, 6]
        invalid_inputs = [[np.array([[1.], [2.]]), np.array([[1.]])],
                          [np.array([[1., 2.]]), np.array([[2.], [2.]])],
                          [np.array([[1.], [2.]]), np.eye(2)],
                          [np.eye(1), np.eye(1)]]

        for i in range(len(invalid_input_inds)):
            inputs_new = copy.copy(self.inputs)
            inputs_new[invalid_input_inds[i]] = invalid_inputs[i]
            with pytest.raises(ValueError):
                gen_samples_A_L(*inputs_new)

