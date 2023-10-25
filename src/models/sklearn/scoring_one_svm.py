
from typing import Literal
from numpy.random import RandomState
from sklearn.base import ClassifierMixin
from sklearn.linear_model import SGDOneClassSVM



class ScoringSGDOneClassSVM(SGDOneClassSVM, ClassifierMixin):
    """
    This class was made to inherit from SGDOneClassSVM and ClassifierMixin
    so it would be able to score on a test dataset for using in tuning experiments
    """
    def __init__(
            self,
            nu=0.5,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-3,
            shuffle=True,
            verbose=0,
            random_state=None,
            learning_rate="optimal",
            eta0=0.0,
            power_t=0.5,
            warm_start=False,
            average=False,
        ):
        self.nu = nu
        super(SGDOneClassSVM, self).__init__(
            loss="hinge",
            penalty="l2",
            C=1.0,
            l1_ratio=0,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=0.1,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            warm_start=warm_start,
            average=average,
        )
