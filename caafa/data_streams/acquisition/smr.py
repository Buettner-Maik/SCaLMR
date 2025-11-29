import collections

from caafa.utils import utils
from caafa.data_streams.constants import LABEL_DICT_KEY
#import caafa.data_streams.constants as const
from river.preprocessing.scale import safe_div

from dataclasses import dataclass, field

@dataclass
class _SupervisedMeritRankingOptions():
    label_quality_confidence_modifier: bool = False
    label_quality_completeness_modifier: bool = False
    
    def validate(self):
        pass
        #if self.quality_completeness:
        #    raise NotImplementedError
        #if self.retrain_on_updated_values:
        #    raise NotImplementedError

class SupervisedMeritRanking():
    """Chooses acquisition sets based on their merit (quality).
    A merit (quality) is the importance of a feature divided by its acquisition cost.

    Attributes
    ----------
    n : dict[label:float]
        The current sum of weights. If each passed weight was 1, then this is equal to the number
        of seen observations.

    Examples
    --------

    >>> from river import stats

    >>> X = [-5, -3, -1, 1, 3, 5]
    >>> mean = stats.Mean()
    >>> for x in X:
    ...     mean.update(x)
    ...     print(mean.get())
    -5.0
    -4.0
    -3.0
    -2.0
    -1.0
    0.0

    You can calculate a rolling average by wrapping a `utils.Rolling` around:

    >>> from river import utils

    >>> X = [1, 2, 3, 4, 5, 6]
    >>> rmean = utils.Rolling(stats.Mean(), window_size=2)

    >>> for x in X:
    ...     rmean.update(x)
    ...     print(rmean.get())
    1.0
    1.5
    2.5
    3.5
    4.5
    5.5
    
    """
        
    def __init__(self, feature_importance, feature_selection, acquisition_costs, **kwargs):
        """
        feature_importance must be of type {fname:importance_measure}
        """
        self.feature_importance = feature_importance
        self.feature_selection = feature_selection
        self.acquisition_costs = acquisition_costs

        self.options = _SupervisedMeritRankingOptions()
        for key, value in kwargs.items():
            if hasattr(self.options, key):
                setattr(self.options, key, value)
            else:
                raise ValueError(f"Undefined option '{key}'")
        self.options.validate()
        
    def get_set(self, x, y, y_conf=1.0):
        missing = utils.get_missing(x)

        qualities = {i: self.feature_importance[i].get() / self.acquisition_costs[i]
                  for i in x
                  if missing[i]}

        if y is None:
            qualities[LABEL_DICT_KEY] = self.feature_importance[LABEL_DICT_KEY].get() / self.acquisition_costs[LABEL_DICT_KEY]

            if self.options.label_quality_confidence_modifier:
                qualities[LABEL_DICT_KEY] *= 1 - y_conf
            
            if self.options.label_quality_completeness_modifier:
                num = 0.0
                denom = 0.0
                for i in x:
                    fimp = self.feature_importance[i].get()
                    num += (missing[i] + 1) % 2 * fimp
                    denom += fimp
                if denom != 0:
                    qualities[LABEL_DICT_KEY] *= num / denom

        return self.feature_selection.transform_one(qualities)

    def get_quality(self, xy, y_conf=1.0):
        if not any(xy):
            return 0
        
        not_missing = utils.get_not_missing(xy)
        qualities = xy.copy()
        for i in xy:
            qualities[i] = self.feature_importance[i].get() / self.acquisition_costs[i] * not_missing[i]

        if self.options.label_quality_confidence_modifier:# and not not_missing[LABEL_DICT_KEY]:
            qualities[LABEL_DICT_KEY] *= 1 - y_conf
        
        if self.options.label_quality_completeness_modifier:
            num = 0.0
            denom = 0.0
            for i in xy:
                if i != LABEL_DICT_KEY:
                    fimp = self.feature_importance[i].get()
                    num += fimp * not_missing[i]
                    denom += fimp
            if denom != 0:
                qualities[LABEL_DICT_KEY] *= num / denom

        return safe_div(sum(qualities.values()), sum(not_missing.values()))
    
    # def get_cost(self, x, y):
    #     not_missing = utils.get_not_missing(x)
    #     costs = x.copy()
    #     for i in x:
    #         costs[i] = self.acquisition_costs[i] * not_missing[i]
    #     costs[const.LABEL_STR_KEY] = self.acquisition_costs[const.LABEL_STR_KEY] * (y != None)

    #     return sum(costs.values())

    def learn_one(self, x, y):
        for f, value in x.items():
            if value is not None:
                self.feature_importance[f].update(value, y)
        self.feature_importance[LABEL_DICT_KEY].update(y, y)
        self.feature_selection.learn_one(x)

    def forget_one(self, x, y):
        for f, value in x.items():
            self.feature_importance[f].revert(value, y)
        self.feature_importance[LABEL_DICT_KEY].revert(y, y)
        self.feature_selection.forget_one(x)

    def __str__(self):
        return "SupervisedMeritRanking"
    
    def __repr__(self):
        return "SupervisedMeritRanking"