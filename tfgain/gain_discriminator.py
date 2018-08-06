import tensorflow as tf

class GAINDiscriminator(object):
    """This class implements disciminator to detect which positions
    of imputed data are observed.
    """
    def __init__(self):
        pass

    def infer(self, xbar, m, b):
        """return probability whether each position are observed.

        Parameters
        ----------
        xhat : tf.Tensor
            result of missing value imputation of x.
        m : tf.Tensor
            mask data indicating missing positions in x by 0/1 flag
            (0=missing, 1=observed ; same size as xhat)
        b : tf.Tensor
            Hint flag data (integer)
            each row has single "1", which is selected at random.
            The other cells have "0" (same size as mhat)

        Returns
        -------
        mhat : tf.Tensor
            result probabilities
        """
        # calculate hint tensor
        h = b * m + 0.5 * (1 - b)

        # MLP
        d = xhat.shape[1]
        out = tf.layers.dense(out, d, activation=tf.tanh, name="dense1")
        out = tf.layers.dense(out, int(d/2), activation=tf.tanh, name="dense2")
        out = tf.layers.dense(out, d, activation=tf.sigmoid, name="dense3")
        return out

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
        eps = 1e-7
        log_loss = m * tf.log(mhat + eps) + (1 - m) * tf.log(1. - mhat + eps)
        loss = tf.reduce_sum((1 - b) * log_loss)

        return loss
