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
        xhat : tf.Tensor of tf.float32
            result of missing value imputation of x.
        m : tf.Tensor of tf.bool
            mask data indicating missing positions in x
            (if True, observed ; same size as xhat)
        b : tf.Tensor of tf.bool
            Hint flag data
            each row has only one True, which is selected at random.
            The other cells are False (same size as mhat)

        Returns
        -------
        mhat : tf.Tensor
            result probabilities
        """
        assert xhat.shape == m.shape == b.shape
        assert xhat.dtype == tf.float32
        assert m.dtype == b.dtype == tf.bool

        # calculate hint tensor
        h = tf.cond(b, tf.cast(m, dtype=tf.float32), 0.5,
                    name="hint")

        # concat imputed tensor and hint
        out = tf.concat([xhat, h], axis=1)

        # MLP
        d = xhat.shape[1]
        out = tf.layers.dense(out, d, activation=tf.tanh,
                              name="dense1")
        out = tf.layers.dense(out, int(d/2), activation=tf.tanh,
                              name="dense2")
        mhat = tf.layers.dense(out, d, activation=tf.sigmoid,
                               name="dense3")
        return mhat

    def loss(self, mhat, m, b):
        """Calculate loss of discriminator

        Parameters
        ----------
        mhat : tf.Tensor of tf.float32
            A prediction result of missing mask of discriminator.
            It contains probability whether it is observed.
        m : tf.Tensor of tf.bool
            mask data indicating missing positions in x
            (if True, observed ; same size as xhat)
        b : tf.Tensor of tf.bool
            Hint flag data.
            each row has only one True, which is selected at random.
            The other cells are False (same size as mhat)

        Returns
        -------
        loss : tf.Tensor (no dimension)
            discriminator loss calculated (negative log loss)
        """
        assert mhat.shape == m.shape == b.shape
        assert mhat.dtype == tf.float32
        assert m.dtype == b.dtype == tf.bool

        eps = 1e-7
        log_loss = tf.cond(m, tf.log(mhat + eps),
                           tf.log(1. - mhat + eps))
        loss = - tf.reduce_sum(tf.cond(b, 0., log_loss))

        return loss
