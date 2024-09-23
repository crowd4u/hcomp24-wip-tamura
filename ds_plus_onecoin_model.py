# The implementation of the DS+OneCoin model 
## This is PoC-level codes, so you should avoid using this code in production-level systems.

# LICENCE NOTICE:
## This code is a modified version of the Dawid-Skene implementation from the crowd-kit.
## We used that codes under the apache-2.0 licence.
##
## Copyright 2020 Crowd-Kit team authors
##
## Original code can be found at: 
## https://github.com/Toloka/crowd-kit/blob/main/crowdkit/aggregation/classification/dawid_skene.py


#%%

__all__ = [
    'DSPlusOneCoin',
]

from typing import List, Optional
from numpy.typing import NDArray

import attr
import numpy as np
import pandas as pd

from crowdkit.aggregation.classification.majority_vote import MajorityVote
from crowdkit.aggregation.base import BaseClassificationAggregator
from crowdkit.aggregation.utils import get_most_probable_labels, named_series_attrib

_EPS = np.float_power(10, -10)


@attr.s
class DSPlusOneCoin(BaseClassificationAggregator):
    """
    This is the extension of the Dawid-Skene model in crowd-kit.
    The inputs and outputs are the same as crowd-kit except for the `smooth` parameter.
    The `smooth` parameter is $S$ in the paper.

    However, there is a important limilation that 
    the `label` columns must be `int` and be consecutive numbers started with `0`,
    for example, `0 or 1` in 2-class tasks, `0, 1, 2, or 3` in 4-class tasks.
    """

    n_iter: int = attr.ib(default=100)
    tol: float = attr.ib(default=1e-5)
    class_priors: Optional[NDArray[np.float64]] = attr.ib(default=None)
    smooth: int = attr.ib(default=1000)

    probas_: Optional[pd.DataFrame] = attr.ib(init=False)
    priors_: Optional[pd.Series] = named_series_attrib(name='prior')
    # labels_
    errors_: Optional[pd.DataFrame] = attr.ib(init=False)
    loss_history_: List[float] = attr.ib(init=False)

    @staticmethod
    def _m_step(data: pd.DataFrame, probas: pd.DataFrame, smooth: int) -> pd.DataFrame:
        """
        We modified the M-step of the Dawid-Skene model in the DS+OneCoin model.
        See details in the paper. 
        """
        joined = data.join(probas, on='task')
        joined.drop(columns=['task'], inplace=True)

        # MLE of worker accuracy and the regulartion
        worker_acc = data.groupby(["task", "worker"])["label"].value_counts().unstack(fill_value=0)
        worker_acc *= probas
        worker_acc["sum"] = 0
        for label in range(len(probas.columns)):
            worker_acc["sum"] += worker_acc[label]
        worker_acc = worker_acc.groupby("worker")["sum"].sum() / data.groupby("worker")["task"].count()
        cm = pd.DataFrame([(1 - worker_acc.values) / (len(probas.columns)-1)] * len(probas.columns), columns=[worker_acc.index]).T
        old_cm = cm.copy()
        for label in range(len(probas.columns)):
            tmp = old_cm.copy()
            tmp[label] = worker_acc.values
            tmp["label"] = label
            if label == 0:
                cm = tmp
            else:
                cm = pd.concat([cm, tmp])
        cm = cm.reset_index().set_index(["worker", "label"]).sort_index()

        errors = joined.groupby(['worker', 'label'], sort=False).sum()
        errors.clip(lower=_EPS, inplace=True)
        errors = errors + smooth  * cm # MAP estimation
        errors /= errors.groupby('worker', sort=False).sum()
        return errors

    @staticmethod
    def _e_step(data: pd.DataFrame, priors: pd.Series, errors: pd.DataFrame) -> pd.DataFrame:
        joined = data.join(np.log2(errors), on=['worker', 'label'])
        joined.drop(columns=['worker', 'label'], inplace=True)
        log_likelihoods = np.log2(priors) + joined.groupby('task', sort=False).sum()
        log_likelihoods.rename_axis('label', axis=1, inplace=True)

        scaled_likelihoods = np.exp2(log_likelihoods.sub(log_likelihoods.max(axis=1), axis=0))
        return scaled_likelihoods.div(scaled_likelihoods.sum(axis=1), axis=0)

    def _evidence_lower_bound(self, data: pd.DataFrame, probas: pd.DataFrame, priors: pd.Series, errors: pd.DataFrame) -> float:
        joined = data.join(np.log(errors), on=['worker', 'label'])

        joined = joined.rename(columns={True: 'True', False: 'False'}, copy=False)
        priors = priors.rename(index={True: 'True', False: 'False'}, copy=False)

        joined.loc[:, priors.index] = joined.loc[:, priors.index].add(np.log(priors))

        joined.set_index(['task', 'worker'], inplace=True)
        joint_expectation = (probas.rename(columns={True: 'True', False: 'False'}) * joined).sum().sum()

        entropy = -(np.log(probas) * probas).sum().sum()
        return float(joint_expectation + entropy)

    def fit(self, data: pd.DataFrame) -> 'DSPlusOneCoin':
        data = data[['task', 'worker', 'label']]

        # Early exit
        if not data.size:
            self.probas_ = pd.DataFrame()
            self.priors_ = pd.Series(dtype=float)
            self.errors_ = pd.DataFrame()
            self.labels_ = pd.Series(dtype=float)
            return self

        # Bayes prior initialization
        if self.class_priors is None:
            self.class_priors = np.ones(len(data['label'].unique())) * 0.0 # no smoothing
        assert len(self.class_priors) == len(data['label'].unique())
        
        # Initialization
        probas = MajorityVote().fit_predict_proba(data)
        tmp_probas = probas.copy()
        tmp_probas += self.class_priors
        tmp_probas = tmp_probas.mean()
        priors = tmp_probas / tmp_probas.sum()
        errors = self._m_step(data, probas, self.smooth)
        loss = -np.inf
        self.loss_history_ = []
        del tmp_probas

        # Updating proba and errors n_iter times
        for _ in range(self.n_iter):
            probas = self._e_step(data, priors, errors)
            tmp_probas = probas.copy()
            tmp_probas += self.class_priors
            tmp_probas = tmp_probas.mean()
            priors = tmp_probas / tmp_probas.sum()
            del tmp_probas
            errors = self._m_step(data, probas, self.smooth)
            new_loss = self._evidence_lower_bound(data, probas, priors, errors) / len(data)
            self.loss_history_.append(new_loss)

            if new_loss - loss < self.tol:
                break
            loss = new_loss

        probas.columns = pd.Index(probas.columns, name='label', dtype=probas.columns.dtype)
        # Saving results
        self.probas_ = probas
        self.priors_ = priors
        self.errors_ = errors
        self.labels_ = get_most_probable_labels(probas)

        return self

    def fit_predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:

        return self.fit(data).probas_

    def fit_predict(self, data: pd.DataFrame) -> pd.Series:

        return self.fit(data).labels_
