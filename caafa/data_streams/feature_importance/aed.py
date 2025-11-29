import itertools
import collections
import numbers
import math

from river import stats
from river.preprocessing.scale import safe_div

from .base import FeatureImportance

#import numpy as np

#from river import utils
#from baseclass

def AED_factory(feature):
    if isinstance(feature, numbers.Number):
        return AED_Numeric()
    else:
        return AED_Nominal()

class AED_Numeric(FeatureImportance):
    """Calculates class specific average euclidean distance,
    !requires that learned and removed values are normalized!

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

    def __init__(self):
        self._means = collections.defaultdict(float)
        self._ns = collections.defaultdict(float)
        self._min = stats.Min()
        self._max = stats.Max()


    def update(self, x, y, w=1.0):
        self._min.update(x)
        self._max.update(x)
        x = safe_div(x - self._min.get(), self._max.get() - self._min.get())

        self._ns[y] += w
        self._means[y] += (w / self._ns[y]) * (x - self._means[y])
    
    def revert(self, x, y, w=1.0):
        x = safe_div(x - self._min.get(), self._max.get() - self._min.get())

        self._ns[y] -= w
        if self._ns[y] < 0:
            raise ValueError("Cannot go below 0")
        elif self._ns[y] == 0:
            self._means[y] = 0.0
        else:
            self._means[y] -= (w / self._ns[y]) * (x - self._means[y])
    
    def get(self):
        return math.sqrt(
            sum(
            (m1 - m2) * (m1 - m2)
            for m1, m2 in itertools.combinations(self._means.values(), 2)
            )
        )
        #return utils.math.minkowski_distance(self._means[y]

class AED_Nominal(FeatureImportance):
    """Calculates class specific average euclidean distance,
    !requires that learned and removed values are normalized!

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

    def __init__(self):
        self._feature_value_label_counts = collections.defaultdict(int)
        self._feature_value_counts = collections.defaultdict(int)
        self._label_counts = collections.defaultdict(int)


    def update(self, x, y):
        self._feature_value_label_counts[(x, y)] += 1
        self._feature_value_counts[x] += 1
        self._label_counts[y] += 1
    
    def revert(self, x, y):
        self._feature_value_label_counts[(x, y)] -= 1
        self._feature_value_counts[x] -= 1
        self._label_counts[y] -= 1
        
        if self._feature_value_label_counts[(x, y)] < 0:
            raise ValueError("Cannot go below 0")
        elif self._feature_value_counts[x] < 0:
            raise ValueError("Cannot go below 0")
        elif self._label_counts[y] < 0:
            raise ValueError("Cannot go below 0")
    
    def get(self):
        return sum(
            sum(
                abs(self._feature_value_label_counts[(x, y1)] / self._label_counts[y1] - self._feature_value_label_counts[(x, y2)] / self._label_counts[y2])
                for x in self._feature_value_counts
            ) / len(self._feature_value_counts.values())
        for y1, y2 in itertools.combinations(self._label_counts, 2)
        )
    
class AED_Numeric_Annealing(FeatureImportance):
    """Calculates class specific average euclidean distance,
    !requires that learned and removed values are normalized!

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

    def __init__(self, alpha=None, limit=None):
        self._means = collections.defaultdict(float)
        self._ns = collections.defaultdict(float)
        self._min = stats.Min()
        self._max = stats.Max()
        
        if alpha is None and limit is None:
            raise ValueError("Either provide an alpha or limit")
        elif alpha is not None and limit is not None:
            raise ValueError("Either provide an alpha or limit, not both")
        if limit is None:
            self._alpha = alpha
            self._limit = 1 / (1 - alpha)
        else:
            self._alpha = 1 - (1 / limit)
            self._limit = limit


    def update(self, x, y, w=1.0):
        self._min.update(x)
        self._max.update(x)
        x = safe_div(x - self._min.get(), self._max.get() - self._min.get())

        self._ns[y] += w
        self._means[y] += (w / self._ns[y]) * (x - self._means[y])
    
    def revert(self, x, y, w=1.0):
        raise NotImplementedError
        x = safe_div(x - self._min.get(), self._max.get() - self._min.get())

        self._ns[y] -= w
        if self._ns[y] < 0:
            raise ValueError("Cannot go below 0")
        elif self._ns[y] == 0:
            self._means[y] = 0.0
        else:
            self._means[y] -= (w / self._ns[y]) * (x - self._means[y])
    
    def get(self):
        return math.sqrt(
            sum(
            (m1 - m2) * (m1 - m2)
            for m1, m2 in itertools.combinations(self._means.values(), 2)
            )
        )
        #return utils.math.minkowski_distance(self._means[y]

class AED_Nominal_Annealing(FeatureImportance):
    """Calculates class specific average euclidean distance,
    !requires that learned and removed values are normalized!

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

    def __init__(self, alpha=None, limit=None):
        self._feature_value_label_counts = collections.defaultdict(int)
        self._feature_value_counts = collections.defaultdict(int)
        self._label_counts = collections.defaultdict(int)

        if alpha is None and limit is None:
            raise ValueError("Either provide an alpha or limit")
        elif alpha is not None and limit is not None:
            raise ValueError("Either provide an alpha or limit, not both")
        if limit is None:
            self._alpha = alpha
            self._limit = 1 / (1 - alpha)
        else:
            self._alpha = 1 - (1 / limit)
            self._limit = limit
        
        # lim_inf = 1 / (1 - alpha)
        # alpha = 1 - (1 / x)

    def update(self, x, y):
        for key in self._feature_value_label_counts:
            self._feature_value_label_counts[key] *= self._alpha
        for key in self._feature_value_counts:
            self._feature_value_counts[key] *= self._alpha
        for key in self._label_counts:
            self._label_counts[key] *= self._alpha

        self._feature_value_label_counts[(x, y)] += 1
        self._feature_value_counts[x] += 1
        self._label_counts[y] += 1
    
    def revert(self, x, y):
        raise NotImplementedError
        # self._feature_value_label_counts[(x, y)] -= 1
        # self._feature_value_counts[x] -= 1
        # self._label_counts[y] -= 1
        
        # if self._feature_value_label_counts[(x, y)] < 0:
        #     raise ValueError("Cannot go below 0")
        # elif self._feature_value_counts[x] < 0:
        #     raise ValueError("Cannot go below 0")
        # elif self._label_counts[y] < 0:
        #     raise ValueError("Cannot go below 0")
    
    def get(self):
        return sum(
            sum(
                abs(self._feature_value_label_counts[(x, y1)] / self._label_counts[y1] - self._feature_value_label_counts[(x, y2)] / self._label_counts[y2])
                for x in self._feature_value_counts
            ) / len(self._feature_value_counts.values())
        for y1, y2 in itertools.combinations(self._label_counts, 2)
        )