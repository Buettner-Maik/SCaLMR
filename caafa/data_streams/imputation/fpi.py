import numbers
import operator
import math
from collections import defaultdict, Counter

from river import metrics
from river.metrics.base import Metric
from river import stats
from river.stats.base import Statistic

from river import linear_model
from river.utils.math import softmax

#from river.preprocessing import StatImputer
from .base import Imputer

def safe_div(x, y):
    return 0.0 if y == 0.0 else x / y

class _NumToNum():
    def __init__(self):
        self.in_var = stats.Var()
        self.cov = stats.Cov()

    def learn_one(self, x, y):
        #next(iter(x.values()))
        for value in x.values():
            self.in_var.update(value)
            self.cov.update(value, y)

    def predict_one(self, x):
        beta_1 = safe_div(self.cov.get(), self.in_var.get())
        beta_0 = self.cov.mean_y.get() - self.cov.mean_x.get() * beta_1
        for value in x.values():
            return value * beta_1 + beta_0

class _CatToNum():
    def __init__(self):
        self.means = defaultdict(stats.Mean)        
    
    def learn_one(self, x, y):
        for value in x.values():
            self.means[value].update(y)
    
    def predict_one(self, x):
        for value in x.values():
            return self.means[value].get()

class _CatToCat():
    def __init__(self):
        self.feature_counts = defaultdict(Counter)
        self._last = None
    
    def learn_one(self, x, y):
        for value in x.values():
            self.feature_counts[value].update({y: 1})
            #self._last = y

    def predict_one(self, x):
        for value in x.values():
            most_common = self.feature_counts[value].most_common(1)
            return self._last if not any(most_common) else most_common[0][0]

class _NumToCat():
    def __init__(self):
        self.means = defaultdict(stats.Mean)
        self._last = None

    def learn_one(self, x, y):
        for value in x.values():
            self.means[y].update(value)
            #self._last = y
    
    def predict_one(self, x):
        if not any(self.means):
            return self._last
        for value in x.values():
            abs_dists = {y: abs(mean.get() - value) for y, mean in self.means.items()}
            return min(abs_dists, key=abs_dists.get)

class _PrevImp():
    def __init__(self):
        self.latest = None

    def learn_one(self, x, y):
        self.latest = y
    
    def predict_one(self, x):
        return self.latest

class FeaturePairImputer(Imputer):
    """
    Models -> [(None, f   )] -> Previous Imputer
           -> [(f   , f   )] -> Simple Imputer
           -> [(f   , g   )] -> Pair Imputer
    """
    Eps = 1 / 1024

    def __init__(self, strategy="best", include_previous_imputer=False,
                 
                 num_simple:Statistic=stats.Mean, cat_simple:Statistic=stats.Mode,

                 num_to_num_model=_NumToNum, cat_to_num_model=_CatToNum, 
                 num_to_cat_model=_NumToCat, cat_to_cat_model=_CatToCat,

                 num_metric:Metric=metrics.RMSE, cat_metric:Metric=metrics.Accuracy):
        self.include_previous_imputer = include_previous_imputer
        
        self._num_simple = num_simple
        self._cat_simple = cat_simple
        
        self._num_to_num_model = num_to_num_model
        self._num_to_cat_model = num_to_cat_model
        self._cat_to_num_model = cat_to_num_model
        self._cat_to_cat_model = cat_to_cat_model

        self._num_metric = num_metric
        self._cat_metric = cat_metric
        
        self.strategy = strategy

        self._models = {}
        self._metrics = {}
        self._feature_numeric = {}

        self._i = 0

    def _get_new_metric(self, f):
        if self._feature_numeric[f]:
            return self._num_metric()
        else:
            return self._cat_metric()

    def _get_new_model(self, f1, f2):
        if f1 == f2:
            if self._feature_numeric[f1]:
                return self._num_simple()
            else:
                return self._cat_simple()
            
        f1n = self._feature_numeric[f1]
        f2n = self._feature_numeric[f2]

        if f1n and f2n:
            return self._num_to_num_model()
        elif f1n and not f2n:
            return self._num_to_cat_model()
        elif not f1n and f2n:
            return self._cat_to_num_model()
        elif not f1n and not f2n:
            return self._cat_to_cat_model()

    def _learn_one_prev_imp(self, x):
        for f, v in x.items():
            if v is None: continue
            if (None, f) not in self._models:
                self._models[(None, f)] = _PrevImp()
                self._metrics[(None, f)] = self._get_new_metric(f)

            model = self._models[(None, f)]

            y_pred = model.predict_one({f:v})
            if y_pred is not None:
                self._metrics[(None, f)].update(v, y_pred)
            model.learn_one(None, v)

    def learn_one(self, x):
        for f1, v1 in x.items():
            if v1 is None: continue

            for f2, v2 in x.items():
                if v2 is None: continue

                # New feature
                if f2 not in self._feature_numeric:
                    self._feature_numeric[f2] = isinstance(v2, numbers.Number)

                # New feature
                if (f1, f2) not in self._models:
                    self._models[(f1, f2)] = self._get_new_model(f1, f2)
                    #self._metrics[f2][f1] = self._get_new_metric(f2)
                    self._metrics[(f1, f2)] = self._get_new_metric(f2)

                model = self._models[(f1, f2)]
                fv1 = {f1: v1}
                if hasattr(model, "predict_one"):
                    y_pred = model.predict_one(fv1)
                else:
                    y_pred = model.get()
                if y_pred is not None:
                    self._metrics[(f1, f2)].update(v2, y_pred)
                if hasattr(model, "learn_one"):
                    model.learn_one(fv1, v2)
                else:
                    model.update(v2)

                # if f1 == f2: # Simple model
                #     y_pred = self._models[(f1, f2)].get()
                #     self._metrics[(f1, f2)].update(v2, y_pred)
                #     self._models[(f1, f2)].update(v1)
                # else: # Else
                    
                #     y_pred = self._models[(f1, f2)].predict_one(fv1)
                #     #self._metrics[f2][f1].update(v2, y_pred)
                #     self._metrics[(f1, f2)].update(v2, y_pred)
                #     self._models[(f1, f2)].learn_one(fv1, v2)
        if self.include_previous_imputer:
            self._learn_one_prev_imp(x)

    def _get_prediction_weights(self, f, preds):
        # Let's assume these are proportions and 1 (100%) is maximum for the metric
        if self._metrics[(f, f)].bigger_is_better:
            weights = {i:1 / (1 - preds[i][0] + FeaturePairImputer.Eps) for i in preds}
        else:
            weights = {i:1 / (preds[i][0] + FeaturePairImputer.Eps) for i in preds}
        return softmax(weights)

    def transform_one(self, x):
        new_x = x.copy()
        for f2, v2 in x.items():
            if v2 is not None: continue
            if f2 not in self._feature_numeric: continue

            v2_preds = {f1:(self._metrics[(f1, f2)].get(), self._models[(f1, f2)].predict_one({f1: v1}))
                        for f1, v1 in x.items()
                        if v1 is not None and (f1, f2) in self._models}
            # Remove models that have no answer
            v2_preds = {key: value for key, value in v2_preds.items() if value[1] is not None}
            # Add simple model
            v2_preds[f2] = (self._metrics[(f2, f2)].get(), self._models[(f2, f2)].get())
            # Add previous imputer
            if self.include_previous_imputer:
                v2_preds[None] = (self._metrics[(None, f2)].get(), self._models[(None, f2)].predict_one(None))
            
            if self.strategy == "best":
                if self._metrics[(f2, f2)].bigger_is_better:
                    new_x[f2] = max(v2_preds.values(), key=operator.itemgetter(0))[1]
                else:
                    new_x[f2] = min(v2_preds.values(), key=operator.itemgetter(0))[1]
            elif self.strategy == "weighted":
                weights = self._get_prediction_weights(f2, v2_preds)

                if self._feature_numeric[f2]:
                    new_x[f2] = sum([weights[f1] * value for f1, (_, value) in v2_preds.items()])
                else:
                    votes = Counter()
                    for f1, (_, vote) in v2_preds.items():
                        votes[vote] += weights[f1]
                    new_x[f2] = votes.most_common(1)[0][0]

        return new_x



class _CatCatToNum():
    def __init__(self):
        self.means = defaultdict(stats.Mean)        
    
    def learn_one(self, x, y):
        v1, v2 = x.values()
        self.means[(v1, v2)].update(y)
    
    def predict_one(self, x):
        v1, v2 = x.values()
        if (v1, v2) not in self.means: return None
        return self.means[(v1, v2)].get()

class _CatCatToCat():
    def __init__(self):
        self.feature_counts = defaultdict(Counter)
        self._last = None
    
    def learn_one(self, x, y):
        v1, v2 = x.values()
        self.feature_counts[(v1, v2)].update({y: 1})
        #self._last = y

    def predict_one(self, x):
        v1, v2 = x.values()
        most_common = self.feature_counts[(v1, v2)].most_common(1)
        return self._last if not any(most_common) else most_common[0][0]

def _identify_features(x):
    for f, v in x.items():
        if isinstance(v, numbers.Number):
            num_f = f
        else:
            cat_f = f
    return num_f, cat_f

class _NumNumToNum():
    def __init__(self):
        # self.in_mean1 = stats.Mean()
        # self.in_mean2 = stats.Mean()
        # self.out_mean = stats.Mean()

        self.in_var1 = stats.Var()
        self.in_var2 = stats.Var()
        self.in_cov = stats.Cov()
        self.out_cov1 = stats.Cov()
        self.out_cov2 = stats.Cov()

    def learn_one(self, x, y):
        v1, v2 = x.values()
        # self.in_mean1.update(v1)
        # self.in_mean2.update(v2)
        # self.out_mean.update(y)
        # v1_ = v1 - self.in_mean1.get()
        # v2_ = v2 - self.in_mean2.get()
        # y_ = y - self.out_mean.get()

        self.in_var1.update(v1)
        self.in_var2.update(v2)
        self.in_cov.update(v1, v2)
        self.out_cov1.update(v1, y)
        self.out_cov2.update(v2, y)

    def predict_one(self, x):
        v1, v2 = x.values()
        delta = self.in_var1.get() * self.in_var2.get() - self.in_cov.get() * self.in_cov.get()
        if delta == 0.0:
            beta_1 = 0.0
            beta_2 = 0.0
            beta_0 = self.out_cov1.mean_y.get()
        else:
            beta_1 = (self.in_var2.get() * self.out_cov1.get() - self.in_cov.get() * self.out_cov2.get()) / delta
            beta_2 = (self.in_var1.get() * self.out_cov2.get() - self.in_cov.get() * self.out_cov1.get()) / delta
            beta_0 = self.out_cov1.mean_y.get() - beta_1 * self.in_var1.mean.get() - beta_2 * self.in_var2.mean.get()
        return beta_0 + beta_1 * v1 + beta_2 * v2

class _NumNumToCat():
    def __init__(self):
        self.means_v1 = defaultdict(stats.Mean)
        self.means_v2 = defaultdict(stats.Mean)
        self._last = None

    def learn_one(self, x, y):
        v1, v2 = x.values()
        self.means_v1[y].update(v1)
        self.means_v2[y].update(v2)
        #self._last = y
    
    def predict_one(self, x):
        if not any(self.means_v1):
            return self._last
        v1, v2 = x.values()
        dists = {y: math.sqrt( (self.means_v1[y].get() - v1) ** 2.0 + (self.means_v2[y].get() - v2) ** 2.0 ) for y in self.means_v1}
        return min(dists, key=dists.get)

class _NumCatToCat():
    def __init__(self):
        self.means = {} #defaultdict(stats.Mean)
        self._num_f = None
        self._cat_f = None
        self._last = None
    
    def learn_one(self, x, y):
        if not any(self.means):
            self._num_f, self._cat_f = _identify_features(x)
        
        v_num = x[self._num_f]
        v_cat = x[self._cat_f]
        if v_cat not in self.means:
            self.means[v_cat] = defaultdict(stats.Mean)
        self.means[v_cat][y].update(v_num)
        #self._last = y

    def predict_one(self, x):
        if not any(self.means):
            self._num_f, self._cat_f = _identify_features(x)

        v_num = x[self._num_f]
        v_cat = x[self._cat_f]
        if v_cat not in self.means:
            return self._last
        abs_dists = {y: abs(mean.get() - v_num) for y, mean in self.means[v_cat].items()}
        return min(abs_dists, key=abs_dists.get)

class _NumCatToNum():
    def __init__(self):
        self.regressors = defaultdict(_NumToNum)

    def learn_one(self, x, y):
        if not any(self.regressors):
            self._num_f, self._cat_f = _identify_features(x)

        v_num = x[self._num_f]
        v_cat = x[self._cat_f]

        self.regressors[v_cat].learn_one({self._num_f:v_num}, y)

    def predict_one(self, x):
        if not any(self.regressors):
            self._num_f, self._cat_f = _identify_features(x)

        v_num = x[self._num_f]
        v_cat = x[self._cat_f]

        return self.regressors[v_cat].predict_one({self._num_f:v_num})

class FeatureTripletImputer(Imputer):
    """
    Models -> [(None, None, f   )] -> Previous Imputer
           -> [(f   , f   , f   )] -> Simple Imputer
           -> [(f   , f   , g   )] -> Pair Imputer
           -> [(f   , g   , h   )] -> Triplet Imputer
    """
    Eps = 1 / 1024

    def __init__(self, strategy="best", include_previous_imputer=False,
                 
                 num_simple:Statistic=stats.Mean, cat_simple:Statistic=stats.Mode,

                 num_to_num_model=_NumToNum, cat_to_num_model=_CatToNum, 
                 num_to_cat_model=_NumToCat, cat_to_cat_model=_CatToCat,
                 
                 numnum_to_num_model=_NumNumToNum, numnum_to_cat_model=_NumNumToCat,
                 catcat_to_num_model=_CatCatToNum, catcat_to_cat_model=_CatCatToCat,
                 numcat_to_num_model=_NumCatToNum, numcat_to_cat_model=_NumCatToCat,

                 num_metric:Metric=metrics.RMSE, cat_metric:Metric=metrics.Accuracy):
        self.include_previous_imputer = include_previous_imputer
        
        self._num_simple = num_simple
        self._cat_simple = cat_simple
        
        self._num_to_num_model = num_to_num_model
        self._num_to_cat_model = num_to_cat_model
        self._cat_to_num_model = cat_to_num_model
        self._cat_to_cat_model = cat_to_cat_model
        
        self._numnum_to_num_model = numnum_to_num_model
        self._numnum_to_cat_model = numnum_to_cat_model
        self._catcat_to_num_model = catcat_to_num_model
        self._catcat_to_cat_model = catcat_to_cat_model
        self._numcat_to_num_model = numcat_to_num_model
        self._numcat_to_cat_model = numcat_to_cat_model

        self._num_metric = num_metric
        self._cat_metric = cat_metric
        
        self.strategy = strategy

        self._models = {}
        self._metrics = {}
        self._feature_numeric = {}

        self._i = 0

    def _get_new_metric(self, f):
        if self._feature_numeric[f]:
            return self._num_metric()
        else:
            return self._cat_metric()

    def _get_new_model(self, fi1, fi2, fo):
        # Simple models
        if fi1 == fi2 == fo:
            if self._feature_numeric[fo]:
                return self._num_simple()
            else:
                return self._cat_simple()

        f1n = self._feature_numeric[fi1]
        f2n = self._feature_numeric[fi2]
        fon = self._feature_numeric[fo]

        # 1 to 1 models
        if fi1 == fi2:
            if f1n and fon:
                return self._num_to_num_model()
            elif f1n and not fon:
                return self._num_to_cat_model()
            elif not f1n and fon:
                return self._cat_to_num_model()
            elif not f1n and not fon:
                return self._cat_to_cat_model()
        
        # 2 to 1 models
        if f1n and f2n and fon:
            return self._numnum_to_num_model()
        elif f1n and f2n and not fon:
            return self._numnum_to_cat_model()
        elif not f1n and not f2n and fon:
            return self._catcat_to_num_model()
        elif not f1n and not f2n and not fon:
            return self._catcat_to_cat_model()
        elif fon:
            return self._numcat_to_num_model()
        elif not fon:
            return self._numcat_to_cat_model()

    def _learn_one_prev_imp(self, x):
        for f, v in x.items():
            if v is None: continue
            if (None, None, f) not in self._models:
                self._models[(None, None, f)] = _PrevImp()
                self._metrics[(None, None, f)] = self._get_new_metric(f)

            model = self._models[(None, None, f)]

            y_pred = model.predict_one({f:v})
            if y_pred is not None:
                self._metrics[(None, None, f)].update(v, y_pred)
            model.learn_one(None, v)

    def learn_one(self, x):
        # 1 to 1 and simple models
        for fi, vi in x.items():
            if vi is None: continue

            for fo, vo in x.items():
                if vo is None: continue

                # New feature
                if fo not in self._feature_numeric:
                    self._feature_numeric[fo] = isinstance(vo, numbers.Number)

                # New feature pair
                if (fi, fi, fo) not in self._models:
                    self._models[(fi, fi, fo)] = self._get_new_model(fi, fi, fo)
                    self._metrics[(fi, fi, fo)] = self._get_new_metric(fo)

                model = self._models[(fi, fi, fo)]
                fvi = {fi: vi}
                if hasattr(model, "predict_one"):
                    y_pred = model.predict_one(fvi)
                else:
                    y_pred = model.get()
                if y_pred is not None:
                    self._metrics[(fi, fi, fo)].update(vo, y_pred)
                if hasattr(model, "learn_one"):
                    model.learn_one(fvi, vo)
                else:
                    model.update(vo)

        # 2 to 1 models
        already_in = set()
        for fi1, vi1 in x.items():
            if vi1 is None: continue
            already_in.add(fi1)

            for fi2, vi2 in x.items():
                if vi2 is None: continue
                if fi2 in already_in: continue

                for fo, vo in x.items():
                    if vo is None: continue
                    if fi1 == fo or fi2 == fo: continue

                    # New feature triplet
                    if (fi1, fi2, fo) not in self._models:
                        self._models[(fi1, fi2, fo)] = self._get_new_model(fi1, fi2, fo)
                        self._metrics[(fi1, fi2, fo)] = self._get_new_metric(fo)

                    model = self._models[(fi1, fi2, fo)]
                    fvi = {fi1: vi1, fi2: vi2}
                    y_pred = model.predict_one(fvi)
                    if y_pred is not None:
                        self._metrics[(fi1, fi2, fo)].update(vo, y_pred)
                    model.learn_one(fvi, vo)

        if self.include_previous_imputer:
            self._learn_one_prev_imp(x)


    def _get_prediction_weights(self, f, preds):
        # Let's assume these are proportions and 1 (100%) is maximum for the metric
        if self._metrics[(f, f, f)].bigger_is_better:
            weights = {i:1 / (1 - preds[i][0] + FeaturePairImputer.Eps) for i in preds}
        else:
            weights = {i:1 / (preds[i][0] + FeaturePairImputer.Eps) for i in preds}
        return softmax(weights)

    def transform_one(self, x):
        # self._i += 1
        # if self._i == 1000:
        #     self._i -= 1
        #     pass

        new_x = x.copy()
        for fo, vo in x.items():
            if vo is not None: continue
            if fo not in self._feature_numeric: continue

            # 1 to 1 models
            vo_preds = {(f1, f1):(self._metrics[(f1, f1, fo)].get(), self._models[(f1, f1, fo)].predict_one({f1: v1}))
                        for f1, v1 in x.items()
                        if v1 is not None and (f1, f1, fo) in self._models}
            # Add 2 to 1 models
            vo_preds2 = {(f1, f2):(self._metrics[(f1, f2, fo)].get(), self._models[(f1, f2, fo)].predict_one({f1: v1, f2: v2}))
                         for f1, v1 in x.items()
                         for f2, v2 in x.items()
                         if v1 is not None and v2 is not None and f1 is not f2 and (f1, f2, fo) in self._models}
            vo_preds = vo_preds | vo_preds2
            # Remove models that have no answer
            vo_preds = {key: value for key, value in vo_preds.items() if value[1] is not None}
            # Add simple model
            vo_preds[(fo, fo)] = (self._metrics[(fo, fo, fo)].get(), self._models[(fo, fo, fo)].get())
            # Add previous imputer
            if self.include_previous_imputer:
                vo_preds[(None, None)] = (self._metrics[(None, None, fo)].get(), self._models[(None, None, fo)].predict_one(None))

            if self.strategy == "best":
                if self._metrics[(fo, fo, fo)].bigger_is_better:
                    new_x[fo] = max(vo_preds.values(), key=operator.itemgetter(0))[1]
                else:
                    new_x[fo] = min(vo_preds.values(), key=operator.itemgetter(0))[1]
            elif self.strategy == "weighted":
                weights = self._get_prediction_weights(fo, vo_preds)

                if self._feature_numeric[fo]:
                    new_x[fo] = sum([weights[f12] * value for f12, (_, value) in vo_preds.items()])
                else:
                    votes = Counter()
                    for f12, (_, vote) in vo_preds.items():
                        votes[vote] += weights[f12]
                    new_x[fo] = votes.most_common(1)[0][0]

        return new_x