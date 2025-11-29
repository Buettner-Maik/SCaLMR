from river import metrics
from river import compose
from river import stats

from ...utils import utils
from ..constants import LABEL_DICT_KEY
#import constants as const

from sortedcontainers import SortedDict
from collections import defaultdict, Counter, deque

import math
import time
import datetime as dt
#from types import SimpleNamespace
from dataclasses import dataclass, field

#miss_transformer = compose.FuncTransformer(get_missing)

@dataclass
class _FrameworkOptions():
    unsupervised_task_learner: bool = False
    unsupervised_task_learner_confidence_threshold: float = 0.95
    unsupervised_acquisition: bool = False
    unsupervised_acquisition_confidence_threshold: float = 0.95
    """The acquisition strategy will use imputed values to train its method on
    """
    imputed_acquisition: bool = True
    """A true label received before or after the acquisition will override the task learner's prediction
    If set to true, labels already available or made available will also be evaluated on and will influence the confusion matrix' results
    """
    immediate_label_overrides_prediction: bool = False

    eval_imputed_features: dict = field(default_factory=dict)

    retrain_on_updated_values: bool = False # NOT IMPLEMENTED
    
    def validate(self):
        if self.retrain_on_updated_values:
            raise NotImplementedError

class Framework():
    def __init__(self, 
                 stream,
                 miss_mask,
                 costs, 
                 preproc_pipe, 
                 acquisition_strategy, 
                 decision_maker,
                 oracle,
                 
                 budget_gain, 
                 
                 impute_pipe, 
                 task_learner,
                 
                 **kwargs):
        self.stream = stream
        self.miss_mask = miss_mask
        self.costs = costs
        self.preproc_pipe = preproc_pipe
        self.acquisition_strategy = acquisition_strategy
        self.decision_maker = decision_maker
        self.oracle = oracle
        
        self.budget_gain = budget_gain

        self.impute_pipe = impute_pipe
        self.task_learner = task_learner

        self.options = _FrameworkOptions()
        for key, value in kwargs.items():
            if hasattr(self.options, key):
                setattr(self.options, key, value)
            else:
                raise ValueError(f"Undefined option '{key}'")
        self.options.validate() 
        # self.options = SimpleNamespace(
        #     learn_by_instance=False
        # )

    def _get_dynamic_budget_threshold(self, penalty_step_size=1/32, penalty_gain=1, default=1.0):
        if self.expected_query_costs.get() == 0 or self.budget_given == 0: return default

        expected_threshold = self.budget_gain / self.expected_query_costs.get()
        used_budget_ratio = self.budget_spent / self.budget_given
        if used_budget_ratio == 0: return default
        new_threshold = expected_threshold / used_budget_ratio
        over_budget_rate = 1 + (math.floor((used_budget_ratio - 1) / penalty_step_size) + 1) * penalty_gain
        # e.g. [1, 1+p) -> 1/2, [1+p, 1+2p) -> 1/3, ...

        if used_budget_ratio > 1: new_threshold /= over_budget_rate
        return min(new_threshold, 1.0)

    def generate_stats(self):
        import caafa.data_streams.constants as const
        
        result = {
            const.STATKEY_MODEL_CM: self.cm.data,
            const.STATKEY_MODEL_ACC: self.acc.get(),

            const.STATKEY_BUDGET_SPENT: self.budget_spent,
            const.STATKEY_BUDGET_GIVEN: self.budget_given,

            const.STATKEY_POSITIVE_DECISIONS: self.positive_decisions,
            const.STATKEY_DECISION_MAKER_BUDGET_THRESHOLD: self.decision_maker.budget_threshold,

            const.STATKEY_MEAN_INST_QUALITY: self.mean_inst_quality.get(),
            const.STATKEY_MEAN_QUALITY_GAIN: self.mean_quality_gain.get(),
            const.STATKEY_ACQUISITIONS: self.values_queried,
            const.STATKEY_FEATURE_IMPORTANCES: {f:fi_aed.get() for f, fi_aed in self.acquisition_strategy.feature_importance.items()},
            
            const.STATKEY_IMPUTATIONS: self.imputed_features,
            const.STATKEY_IMPUTATION_PERF: {f:perf.get() for f, perf in self.feature_evaluators.items()},

            const.STATKEY_CHANGED_PREDICTION_AFTER_QUERY: self.query_changed_prediction,
            const.STATKEY_CORRECT_PREDICTION_AFTER_QUERY: self.changed_prediction_now_correct,
            const.STATKEY_QUERIES_ASKED: self.queries_posed,
            #const.STATKEY_QUERIES_ANSWERED: self.queries_answered,
            
            const.STATKEY_COMPUTATION_TIME: dt.timedelta(seconds=self.time_end - self.time_start)
        }
        
        return result
        #pd.DataFrame(pd.Series(data)).transpose().to_csv(filepath, sep='\t')

    def _print_stats(self):
        time_now = time.perf_counter()
        print(self.cm)
        print(self.acc)
        print(f"Budgets {self.budget_spent}/{self.budget_given}")
        print("Budget Threshold", self.decision_maker.budget_threshold)
        print("Expected Query Size and Cost", self.expected_query_set_size, self.expected_query_costs)
        print("Mean Instance Quality and Gain", self.mean_inst_quality, self.mean_quality_gain)
        print("Positive Decisions", self.positive_decisions)
        #print(f"Queries {self.queries_answered}/{self.queries_posed}")
        print(f"Queries changed prediction (for the better) {self.query_changed_prediction}({self.changed_prediction_now_correct})")
        print("Time taken", dt.timedelta(seconds=time_now - self.time_start))
        print("Acquisitions", dict(sorted(self.values_queried.items(), key=lambda item:item[1], reverse=True)))
        imps = {f:fi_aed.get() for f, fi_aed in self.acquisition_strategy.feature_importance.items()}
        print("Importances", dict(sorted(imps.items(), key=lambda item:item[1], reverse=True)))

        if any(self.options.eval_imputed_features):
            print("imputed features", self.imputed_features)
            print("feauter imputation performance", self.feature_evaluators)


    def _imbue_misses(self, x, y, miss_mask):
        x = x.copy()
        for f, missing in miss_mask.items():
            if f in x and missing:
                x[f] = None
        y = None if miss_mask[LABEL_DICT_KEY] else y
        return x, y
    
    def _get_cost(self, acq):
        return sum([cost for f, cost in self.costs.items() if f in acq])

    def initialize(self, initial_budget, pre_train):
        self.budget_init = initial_budget
        self.pre_train = pre_train

        self.time_start = time.perf_counter()
        self.budget_given = initial_budget
        self.budget_spent = 0.0
        self.cm = metrics.ConfusionMatrix()
        self.acc = metrics.Accuracy(self.cm)
        self.mean_inst_quality = stats.Mean()
        self.mean_quality_gain = stats.Mean()
        self.expected_query_costs = stats.Mean()
        self.expected_query_set_size = stats.Mean()

        self.positive_decisions = 0
        self.queries_posed = 0
        self.values_queried = defaultdict(int)
        #self.queries_answered = 0
        #self.values_answered = defaultdict(int)

        self.query_changed_prediction = 0
        self.changed_prediction_now_correct = 0

        if any(self.options.eval_imputed_features):
            self.feature_evaluators = {f: metrics.RMSE() for f, is_num in self.options.eval_imputed_features.items()
                                       if is_num} | {f: metrics.Accuracy() for f, is_num in self.options.eval_imputed_features.items()
                                                     if not is_num}
            self.imputed_features = Counter()

        if pre_train <= 0:
            return 0
        for index, ((X_true, y_true), miss_mask) in enumerate(zip(self.stream, self.miss_mask)):
            # Pre steps ---------------------------
            X_raw, y_raw = self._imbue_misses(X_true, y_true, miss_mask)
            true_label_given = y_raw is not None

            # Processing --------------------------
            X_imp = self.impute_pipe.transform_one(X_raw)

            X_pro = self.preproc_pipe.transform_one(X_imp)

            if true_label_given and self.options.immediate_label_overrides_prediction:
                y_pred = y_raw
                y_conf = 1
            else:
                y_prob = self.task_learner.predict_proba_one(X_pro)
                y_conf = y_prob[max(y_prob, key=y_prob.get)] if any(y_prob) else 0
                y_pred = max(y_prob, key=y_prob.get) if any(y_prob) else None

            # Training ----------------------------
            self.impute_pipe.learn_one(X_raw)
            self.preproc_pipe.learn_one(X_imp)
            
            if true_label_given:
                self.task_learner.learn_one(X_pro, y_true)
            elif self.options.unsupervised_task_learner and y_conf >= self.options.unsupervised_task_learner_confidence_threshold:
                self.task_learner.learn_one(X_pro, y_pred)

            if true_label_given:
                if self.options.imputed_acquisition:
                    self.acquisition_strategy.learn_one(X_imp, y_true)
                else:
                    self.acquisition_strategy.learn_one(X_raw, y_true)
            elif self.options.unsupervised_acquisition and y_conf >= self.options.unsupervised_acquisition_confidence_threshold:
                if self.options.imputed_acquisition:
                    self.acquisition_strategy.learn_one(X_imp, y_pred)
                else:
                    self.acquisition_strategy.learn_one(X_raw, y_pred)

            if index + 1 == pre_train:
                break

        return index + 1

    def start_run(self, initial_budget=0.0, pre_training_size=10, graph_filepath=None):
        index = self.initialize(initial_budget=initial_budget, pre_train=pre_training_size)

        for index, ((X_true, y_true), miss_mask) in enumerate(zip(self.stream, self.miss_mask), index):

            # Debug early stops -------------------
            # if index >= 1000:
            #     break
            # if index % 1000 == 0:
            #     self._print_stats()
                # if hasattr(self.task_learner, "draw"):
                #     _ = self.task_learner.draw()
                #     _.view(filename=f"{index}.gv")
                # raise

            # Pre steps ---------------------------
            self.decision_maker.budget_threshold = self._get_dynamic_budget_threshold()
            self.budget_given += self.budget_gain
            X_raw, y_raw = self._imbue_misses(X_true, y_true, miss_mask)
            true_label_given = y_raw is not None

            # Processing --------------------------
            X_imp = self.impute_pipe.transform_one(X_raw)

            X_pro = self.preproc_pipe.transform_one(X_imp)

            if true_label_given and self.options.immediate_label_overrides_prediction:
                y_pred = y_raw
                y_conf = 1
            else:
                y_prob = self.task_learner.predict_proba_one(X_pro)
                y_conf = y_prob[max(y_prob, key=y_prob.get)] if any(y_prob) else 0
                y_pred = max(y_prob, key=y_prob.get) if any(y_prob) else None

            # Acquisition -------------------------
            acq_set = self.acquisition_strategy.get_set(X_raw, y_raw, y_conf)
            
            if any(acq_set):
                acq_cost = self._get_cost(acq_set)
                self.expected_query_set_size.update(len(acq_set))
                self.expected_query_costs.update(acq_cost)
                
                Xy_raw = X_raw | {LABEL_DICT_KEY:y_raw}
                quality_before = self.acquisition_strategy.get_quality(Xy_raw, y_conf)
                Xy_acq = X_raw | {LABEL_DICT_KEY:y_raw} | acq_set
                quality_after = self.acquisition_strategy.get_quality(Xy_acq, y_conf)
                quality_gain = quality_after - quality_before

                self.mean_inst_quality.update(quality_before)
                self.mean_quality_gain.update(quality_gain)

                # Deciding acquisition ----------------
                self.decision_maker.learn_one(quality_gain)
                will_acquire = self.decision_maker.transform_one(quality_gain)
                #print(self.decision_maker.values[:5])
                #print(self.decision_maker.values_queue)

                if will_acquire:
                    self.positive_decisions += 1
                    self.budget_spent += acq_cost
                    for f in acq_set:
                        self.queries_posed += 1
                        self.values_queried[f] += 1
                        if f is LABEL_DICT_KEY:
                            self.oracle.query_one(index, f, y_true, (X_true, y_pred))
                        else:
                            self.oracle.query_one(index, f, X_true[f], (X_raw[f], y_pred))
                else:
                    pass
            
            # Resolving delayed answers -----------
            answered_queries = self.oracle.retrieve_many(index)
            immediate_answer = False
            for (answered_i, answered_f), (answered_true_value, answered_old_value) in answered_queries.items():
                if answered_i == index:
                    immediate_answer = True
                    if answered_f is LABEL_DICT_KEY:
                        true_label_given = True
                        if self.options.immediate_label_overrides_prediction and y_pred != answered_true_value:
                            y_pred = answered_true_value
                            self.query_changed_prediction += 1
                            self.changed_prediction_now_correct += 1
                    else:
                        X_raw[answered_f] = answered_true_value
                elif self.options.retrain_on_updated_values: 
                    raise NotImplementedError
                    # We can only unlearn what we've learned in the first place
                    x_old, y_old = answered_old_value
                    self.acquisition_strategy.forget_one(x_old, y_old)

            # Evaluation --------------------------
            if immediate_answer:
                # If some information was updated about X_raw, everything has to be newly transformed
                X_imp = self.impute_pipe.transform_one(X_raw)
                X_pro = self.preproc_pipe.transform_one(X_imp)
                if not true_label_given or not self.options.immediate_label_overrides_prediction:
                    y_pred_old = y_pred
                    y_pred = self.task_learner.predict_one(X_pro)
                    if y_pred_old != y_pred:
                        self.query_changed_prediction += 1
                        if y_pred == y_true:
                            self.changed_prediction_now_correct += 1

            if y_pred is not None:
                self.cm.update(y_true, y_pred)
            if any(self.options.eval_imputed_features):
                for f, evaluator in self.feature_evaluators.items():
                    if X_raw[f] is not None: continue
                    evaluator.update(X_true[f], X_imp[f])
                    self.imputed_features[f] += 1

            # Training ----------------------------
            self.impute_pipe.learn_one(X_raw)
            self.preproc_pipe.learn_one(X_imp)

            if true_label_given:
                self.task_learner.learn_one(X_pro, y_true)
            elif self.options.unsupervised_task_learner and y_conf >= self.options.unsupervised_task_learner_confidence_threshold:
                self.task_learner.learn_one(X_pro, y_pred)

            if true_label_given:
                if self.options.imputed_acquisition:
                    self.acquisition_strategy.learn_one(X_imp, y_true)
                else:
                    self.acquisition_strategy.learn_one(X_raw, y_true)
            elif self.options.unsupervised_acquisition and y_conf >= self.options.unsupervised_acquisition_confidence_threshold:
                if self.options.imputed_acquisition:
                    self.acquisition_strategy.learn_one(X_imp, y_pred)
                else:
                    self.acquisition_strategy.learn_one(X_raw, y_pred)
        
        self.time_end = time.perf_counter()
        results = self.generate_stats()
        # self._print_stats()
        # if hasattr(self.task_learner, "draw") and index < 50000:
        #     from graphviz import Digraph
        #     _:Digraph = self.task_learner.draw()
        #     _.render(filename=graph_filepath, format="pdf")
        return results