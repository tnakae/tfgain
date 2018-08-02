import numpy as np
import pandas as pd
import tensorflow as tf

from .gain_discriminator import GAINDiscriminator
from .gain_generator import GAINGenerator
from .mask import MissingMask

class GAIN(object):
    """Implementation of next paper:
    Jinsun Yoon+ (ICML2018) GAIN : Missing Data Imputation
    using Generative Adversarial Nets

    GAIN algorithm is composed of GAN-like structure in which
    generator and discriminator are included but different from
    standard one.
    In GAIN, generator imputes missing value with proper value,
    and discriminator infers which values are imputed.

    This implementation uses tensorflow.
    To impute missing value, call "fit" to train at first.
    Then, call "transform" to impute.
    """
    def __init__(self, n_batch=64, alpha=0.1):
        """
        Parameters
        ----------
        n_batch : int
            batch size
        alpha : float
            coefficient of loss of observed compoenents
        """
        self.n_batch = n_batch
        self.alpha = alpha

    def fit(self, x):
        """fit GAIN imputation model against training data x
        with missing value

        Parameters
        ----------
        x : pd.DataFrame
            training data with missing value.
            np.nan and None are accepted as missing values.

        Returns
        -------
        model : GAIN object
            This object
        """
        pass

    def transform(self, x):
        """Impute missing data with proper values.

        Parameters
        ----------
        x : pd.DataFrame
            Target data with missing value.
            np.nan and None are accepted as missing values.

        Returns
        -------
        y : pd.DataFrame
            Data Frame with missing values imputed.
        """
        pass
