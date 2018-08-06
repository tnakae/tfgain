import tensorflow as tf

class GAINGenerator(object):
    """This is class to impute missing value with proper values from
    observed data and missing mask (0/1 flags indicating missing value)
    """
    def __init__(self):
        pass

    def generate(self, x, m, z):
        """Generate candidate values to be imputated.

        Parameters
        ----------
        x : tf.Tensor
            data frame which is target of missing value imputation.
        m : tf.Tensor
            mask data indicating missing positions in x by 0/1 flag
            (0=missing, 1=observed ; same size as x)
        z : tf.Tensor
            data frame each cell of which has random numbers
            to generate imputed values (same size as x)

        Returns
        -------
        xbar : tf.Tensor
            generated data frame which has candidate values
            (even in observed cell)
        """
        assert x.shape == m.shape == z.shape
        out = tf.concat([x, m, z], axis=1, name="concat")
        d = x.shape[1]

        out = tf.layers.dense(out, d, activation=tf.tanh, name="dense1")
        out = tf.layers.dense(out, int(d/2), activation=tf.tanh, name="dense2")
        out = tf.layers.dense(out, d, activation=tf.sigmoid, name="dense3")
        return out

    def impute(self, x, xbar, m):
        """Do missing value imputation. This method uses candidate
        values in xbar (which is generated by generate method)

        Parameters
        ----------
        x : tf.Tensor
            data frame which is target of missing value imputation.
        xbar : tf.Tensor
            data frame which is result of generate method.
            all of missing value of x are imputed by candidate values.
            (same size as x)
        m : tf.Tensor
            mask data indicating missing positions in x by 0/1 flag
            (0=missing, 1=observed ; same size as x)

        Returns
        -------
        xhat : tf.Tensor
            result of missing value imputation of x.
        """
        xhat = x * m + xbar * (1. - m)
        return xhat

    def adversarial_loss(self, mhat, m, b):
        """Calculate adversarial loss. This method compares
        actual missing mask from output of discriminator, and
        uses hint (b).

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
            adversarial loss calculated
        """
        eps = 1e-7
        log_loss = - (1 - m) * tf.log(mhat + eps)
        loss = tf.reduce_sum((1 - b) * log_loss)

        return loss

    def generate_loss(self, x, xbar, m):
        """Calculate generate loss.
        The more x is similar to xbar, the less loss is.

        Parameters
        ----------
        x : tf.Tensor
            data frame which is target of missing value imputation.
        xbar : tf.Tensor
            data frame which is result of generate method.
            all of missing value of x are imputed by candidate values.
            (same size as x)
        m : tf.Tensor
            mask data indicating missing positions in x by 0/1 flag
            (0=missing, 1=observed ; same size as x)

        Returns
        -------
        loss : tf.Tensor (no dimension)
            generate loss calculated
        """
        mse = tf.square(x - xbar)
        loss = tf.reduce_sum(m * mse)

        return loss
