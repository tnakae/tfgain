import tensorflow as tf

class GAINDiscriminator(object):
    """This class implements disciminator to detect which positions
    of imputed data are observed.
    """
    def __init__(self):
        pass

    def infer(self, xbar, b):
        """return probability whether each position are observed.

        Parameters
        ----------
        xhat : tf.Tensor
            result of missing value imputation of x.
        b : tf.Tensor
            Hint flag data (integer)
            each row has single "1", which is selected at random.
            The other cells have "0" (same size as mhat)

        Returns
        -------
        mhat : tf.Tensor
            result probabilities
        """
        pass

    def loss(self, mhat, m, b):
        """calculate loss of discriminator

        Parameters
        ----------
        mhat : tf.Tensor
            A prediction result of missing mask of discriminator.
            It contains probability whether it is observed.
        m : tf.Tensor
            actual missing mask (0/1 flags, same size as mhat)
        b : tf.Tensor
            Hint flag data (integer)
            each row has single "1", which is selected at random.
            The other cells have "0" (same size as mhat)

        Returns
        -------
        loss : tf.Tensor (no dimension)
            discriminator loss calculated
        """
        pass

    def optimize(self, loss):
        """Return optimizer of discriminator.
        This optimizer control variables only in the disctiminator.

        Parameters
        ----------
        loss : tf.Tensor (no dimenstion)
            loss to be minimized

        Returns
        -------
        minimizer : tf.Operation
            minimizer to minimize loss
        """
        pass
