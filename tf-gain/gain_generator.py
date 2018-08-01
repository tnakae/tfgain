"""
"""
class GAINGenerator(object):
    """This is class to impute missing value with proper values from
    observed data and missing mask (0/1 flags indicating missing value)
    """
    def __init__(self):
        pass

    def generate(self, x, m, z):
        """Generate candidates to impute missing values

        Parameters
        ----------
        """
        pass

    def impute(self, x, xbar, m):
        pass

    def adversarial_loss(self, mhat, m, b):
        pass

    def generate_loss(self, x, xbar, m):
        pass

    def optmize(self, loss):
        pass
