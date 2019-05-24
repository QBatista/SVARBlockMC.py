"""
A module implementing the block MC method from Zha (1999).

"""

from jax import random
import jax.numpy as np


def gen_samples_A_L(keys_A_ii_0, keys_C_i, samples_nb, T, S_C_hat, C_hat,
                    X_T_X_inv):
    """
    Generate MC samples for the matrix form of
    :math:`A\left(L\right)y\left(t\right)` in the following structural VAR
    model using Algorithm 1 in Zha (1999):

    .. math::

        A\left(L\right)y\left(t\right)=\epsilon\left(t\right)

    Parameters
    -----------
    keys_A_ii_0 : list
        A list of PRNG keys, one for each block, used to sample 
        :math:`A_{ii}\left(0\right)`.

    keys_C_i : list
        A list of PRNG keys, one for each block, used to sample :math:`C_{i}`.

    samples_nb : scalar(int)
        Number of samples to generate for each block.

    T : scalar(int)
        Length of the time series.

    S_C_hat : list
        A list of arrays containing
        :math:`S_{i}\left(\hat{\boldsymbol{C}}_{i}\right)`, one for each
        block. Each element must be of dimension :math:`m_{i}` by
        :math:`m_{i}`.

    C_hat : list
        A list of arrays containing :math:`\hat{\boldsymbol{C}}_{i}`, one for
        each block. Each element must be of dimension :math:`k_{i}` by
        :math:`m_{i}`.

    X_T_X_inv : list
        A list of arrays containing
        :math:`\left(X'X\right)^{-1}`, one for each block.
        Each element must be of dimension :math:`k_{i}` by :math:`k_{i}`.

    Returns
    -----------
    A_L : ndarray(float)
        An array containing MC samples for the matrix form of
        :math:`A\left(L\right)y\left(t\right)`.

    References
    -----------

    .. [1] Zha, Tao. 1999. Block recursion and structural vector
           autoregressions. Journal of Econometrics 90 (2):291–316.

    """

    n = len(S_C_hat)

    if n != len(keys_A_ii_0):
        raise ValueError("Number of keys for sampling A_ii_0 must match the" +
                         " length of S_C_hat.")

    if n != len(keys_C_i):
        raise ValueError("Number of keys for sampling C_i must match the" + 
                         " length of S_C_hat.")
 
    if n != len(keys_C_i):
        raise ValueError("The length of C_hat must match the length of" + 
                         " S_C_hat.")

    if n != len(keys_C_i):
        raise ValueError("The length of X_T_X_inv must match the length of" +
                         " S_C_hat.")

    A_L = []
    m_i_minus = 0

    for i in range(n):
        # Unpack parameters
        S_i_C_hat_i = S_C_hat[i]
        C_i_hat = C_hat[i]
        X_i_T_X_i_inv = X_T_X_inv[i]

        key_A_ii_0 = keys_A_ii_0[i]
        key_C_i = keys_C_i[i]

        # Check parameters' validity
        _check_valid_params(S_i_C_hat_i, C_i_hat, X_i_T_X_i_inv)

        # Step (a): draw A_ii_0
        A_ii_0, A_ii_0_T_A_ii_0 = gen_samples_A_ii_0(key_A_ii_0, samples_nb,
                                                     T, S_i_C_hat_i)

        # Step (b): draw C_i
        C_i = gen_samples_C_i(key_C_i, samples_nb, C_i_hat, A_ii_0_T_A_ii_0,
                              X_i_T_X_i_inv)

        # Modified step (c): compute A_i(L) (see design notes)
        m_i, k_i = S_i_C_hat_i.shape[0], X_i_T_X_i_inv.shape[0]

        A_L.append(A_ii_0 * (np.eye(m_i, M=k_i, k=m_i_minus) -
                             C_i.reshape((samples_nb, m_i, k_i))))

        m_i_minus += m_i

    return A_L


def gen_samples_A_ii_0(key, samples_nb, T, S_i_C_hat_i):
    """
    FIXME(QBatista): Add documentation

    """

    m_i = S_i_C_hat_i.shape[0]

    shape_param = T / 2 + 1
    scale_param = 2 / S_i_C_hat_i
    size = (samples_nb, m_i, m_i)

    if m_i == 1:
        # Uses https://en.wikipedia.org/wiki/Gamma_distribution#Scaling
        A_ii_0_T_A_ii_0 = scale_param * random.gamma(key, shape_param,
                                                     shape=size)
        A_ii_0 = np.sqrt(A_ii_0_T_A_ii_0)  # because m_i == 1

        return A_ii_0, A_ii_0_T_A_ii_0
    else:
        raise NotImplementedError


def gen_samples_C_i(key, samples_nb, C_i_hat, A_ii_0_T_A_ii_0,
                    X_i_T_X_i_inv):
    """
    FIXME(QBatista): Add documentation

    """
    m_i = C_i_hat.shape[1]
    k_i = X_i_T_X_i_inv.shape[0]

    size = (samples_nb, k_i, m_i)

    if m_i == 1:
        cov = np.kron(np.linalg.inv(A_ii_0_T_A_ii_0), X_i_T_X_i_inv)
        chol_decomp_cov = np.linalg.cholesky(cov)
        C_i = C_i_hat[np.newaxis] + chol_decomp_cov @ random.normal(key, size)

        return C_i
    else:
        raise NotImplementedError


def _check_valid_params(S_i_C_hat_i, C_i_hat, X_i_T_X_i_inv):
    """
    Check the validity of parameters for block sampling.

    """

    if S_i_C_hat_i.shape[0] != S_i_C_hat_i.shape[1]:
        raise ValueError("Each element of S_C_hat must be a square matrix.")

    if S_i_C_hat_i.shape[0] != C_i_hat.shape[1]:
        raise ValueError("The dimensions of elements of S_C_hat and C_hat" +
                         " must match appropriately.")

    if X_i_T_X_i_inv.shape[0] != X_i_T_X_i_inv.shape[1]:
        raise ValueError("Each element of X_T_X_inv must be a square matrix.")

    if X_i_T_X_i_inv.shape[0] != C_i_hat.shape[0]:
        raise ValueError("The dimensions of elements of X_T_X_inv and C_hat must" +
                         " match appropriately.")

