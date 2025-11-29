"""
    Step 1:

    A simple imputation, such as imputing the mean, is performed for every missing value in the dataset. These mean imputations can be thought of as “place holders.”
    Step 2:

    The “place holder” mean imputations for one variable (“var”) are set back to missing.
    Step 3:

    The observed values from the variable “var” in Step 2 are regressed on the other variables in the imputation model, which may or may not consist of all of the variables in the dataset. In other words, “var” is the dependent variable in a regression model and all the other variables are independent variables in the regression model. These regression models operate under the same assumptions that one would make when performing linear, logistic, or Poison regression models outside of the context of imputing missing data.
    Step 4:

    The missing values for “var” are then replaced with predictions (imputations) from the regression model. When “var” is subsequently used as an independent variable in the regression models for other variables, both the observed and these imputed values will be used.
    Step 5:

    Steps 2-4 are then repeated for each variable that has missing data. The cycling through each of the variables constitutes one iteration or “cycle.” At the end of one cycle all of the missing values have been replaced with predictions from regressions that reflect the relationships observed in the data.
    Step 6:

    Steps 2-4 are repeated for a number of cycles, with the imputations being updated at each cycle.
"""

import numbers
import math
from river.base.classifier import Classifier
from river.base.regressor import Regressor
from river import stats
from river import compose, preprocessing
from river.stats.base import Statistic
from river.naive_bayes import GaussianNB
from river import linear_model
from river.base.transformer import Transformer
from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier
from river.tree.hoeffding_tree_regressor import HoeffdingTreeRegressor

#from collections.abc import Callable, Iterable

class MICE():
    def __init__(self, max_cycles:int=5, min_known_ratio_to_learn:float=0.0,
                 
                 num_simple:Statistic=stats.Mean, cat_simple:Statistic=stats.Mode,
                 regression_model:Regressor=HoeffdingTreeRegressor,#linear_model.LinearRegression,
                 classifier_model:Classifier=HoeffdingTreeClassifier):
                 #regression_model:Regressor|Callable[[Iterable], Transformer]=linear_model.LinearRegression, 
                 #classifier_model:Classifier|Callable[[Iterable], Transformer]=GaussianNB):
        """
        :param min_known_ratio_to_learn: How many instances as a ratio of known information vs. all must exist to learn on
                                         MICE will not learn on examples for which the target is unknown.
                                         Simple metrics like mean and mode are always updated.
        """
        
        self.max_cycles = max_cycles
        self.min_known_ratio_to_learn = min_known_ratio_to_learn
        self._min_known_features = -1

        self._models = {} # (set(x - f), f) (only keep max set models for now, (f, f) -> simple)
        self._feature_numeric = {}
        self._num_simple = num_simple
        self._cat_simple = cat_simple
        self._classifier_model = classifier_model
        self._regression_model = regression_model

        self._feature_pipeline = None
        
        self._max_set = None
        self._initialized = False

    def _get_new_model(self, f1, f2):
        if f1 == f2:
            if self._feature_numeric[f2]:
                return self._num_simple()
            else:
                return self._cat_simple()
            
        if self._feature_numeric[f2]:
            return self._regression_model()
        else:
            # if isinstance(self._classifier_model, Classifier):
            return self._classifier_model()
            # else:
            #     return self._classifier_model([f for f in f1 if not self._feature_numeric[f]])

    def _initialize(self, x):
        for f, v in x.items():
            if v is None: raise ValueError(f"The first instance MICE learns on needs to be feature complete. Feature '{f}' was '{v}'.")
            self._feature_numeric[f] = isinstance(v, numbers.Number)
        self._max_set = set(x)
        self._min_known_features = math.ceil(len(x) * self.min_known_ratio_to_learn)

        for f, v in x.items():
            self._models[(f, f)] = self._get_new_model(f, f)
            f_in = frozenset(self._max_set - set([f]))
            self._models[(f_in, f)] = self._get_new_model(f_in, f)

        nominal_features = [f for f, v in self._feature_numeric.items() if not v]
        self._feature_pipeline = ( compose.Discard(*nominal_features) | preprocessing.StandardScaler() ) + ( compose.Select(*nominal_features) | preprocessing.OrdinalEncoder() )

        self._initialized = True

    def learn_one(self, x):
        if not self._initialized:
            self._initialize(x)

        # Only learn complex models if there's enough information to go by
        learn_complex_models = len([f for f, v in x.items() if v is None]) >= self._min_known_features

        if learn_complex_models:
            x_imp = self.transform_one(x)
            self._feature_pipeline.learn_one(x_imp)

        for f, v in x.items():
            if v is None: continue

            # Simple model
            self._models[(f, f)].update(v)

            if learn_complex_models:
                x_in = self._feature_pipeline.transform_one(x_imp)
                f_in = frozenset(self._max_set - set([f]))
                x_in = {f_i: x_in[f_i] for f_i in f_in}
                self._models[(f_in, f)].learn_one(x_in, v)

    def transform_one(self, x):
        if not self._initialized: return x
        # round robin
        x_miss = [f for f, v in x.items() if v is None]

        # initialize with simple models
        x_imp = {f:self._models[(f, f)].get() if v is None else v for f, v in x.items()}

        # round and round it goes
        for i in range(self.max_cycles):
            for f in x_miss:
                x_in = self._feature_pipeline.transform_one(x_imp)
                f_in = frozenset(self._max_set - set([f]))
                x_in = {f_i: x_in[f_i] for f_i in f_in}
                x_imp[f] = self._models[(f_in, f)].predict_one(x_in)
        
        return x_imp
        