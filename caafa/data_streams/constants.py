LABEL_DICT_KEY = "_"

STATKEY_PC_NAME = ("System", "", "PC")
STATKEY_PYTHON_VERSION = ("System", "", "PyVersion")
STATKEY_RUN_DATE = ("System", "", "Date")

STATKEY_UNSUPERVISED_MODEL = ("Framework", "Unsupervised", "Model")
STATKEY_UNSUPERVISED_MODEL_THR = ("Framework", "Unsupervised", "Model_Thr")
STATKEY_UNSUPERVISED_ACQUISITION = ("Framework", "Unsupervised", "Acquisition")
STATKEY_UNSUPERVISED_ACQUISITION_THR = ("Framework", "Unsupervised", "Acquisition_Thr")

STATKEY_IMPUTED_ACQUISITION = ("Framework", "Acquisition", "On_Imputed")
STATKEY_PREDICTION_ACQUISITION = ("Framework", "Acquisition", "Prediction")

STATKEY_BUDGET_INITIAL = ("Framework", "Budget", "Initial")
STATKEY_BUDGET_GAIN = ("Framework", "Budget", "Gain")

STATKEY_TASK_LEARNER = ("Framework", "Module", "Task Learner")
STATKEY_ACQUISITION_STRATEGY = ("Framework", "Module", "Acquisition Strategy")
STATKEY_DECISION_MAKER = ("Framework", "Module", "Decision Maker")
STATKEY_FEATURE_SELECTION = ("Framework", "Module", "Feature Selection")
STATKEY_FEATURE_SELECTION_K = ("Framework", "Module", "FS k")
STATKEY_FEATURE_IMPORTANCE = ("Framework", "Module", "Feature Importance")
STATKEY_LABEL_IMPORTANCE = ("Framework", "Module", "Label Importance")
STATKEY_LABEL_QUAL_CONF_MOD = ("Framework", "Module", "Label Quality Confidence Modifier")
STATKEY_LABEL_QUAL_COMP_MOD = ("Framework", "Module", "Label Quality Completeness Modifier")
STATKEY_IMPUTATION_STRATEGY = ("Framework", "Module", "Imputation Strategy")

STATKEY_DATASET = ("Environment", "Data", "Dataset")
STATKEY_PRE_TRAINING = ("Environment", "Data", "Pre_Train_Instances")
STATKEY_MISS_FEATURES = ("Environment", "Data", "MissFeatures")
STATKEY_MISS_LABELS = ("Environment", "Data", "MissLabels")
STATKEY_COST_FEATURES = ("Environment", "Data", "CostFeatures")
STATKEY_COST_LABEL = ("Environment", "Data", "CostLabels")
STATKEY_DELAY_FEATURES = ("Environment", "Data", "DelayFeatures")
STATKEY_DELAY_LABEL = ("Environment", "Data", "DelayLabels")

STATKEY_MODEL_CM = ("Evaluation", "Task Learner", "Confusions Matrix")
STATKEY_MODEL_ACC = ("Evaluation", "Task Learner", "Accuracy")

STATKEY_BUDGET_SPENT = ("Evaluation", "Budget", "Spent")
STATKEY_BUDGET_GIVEN = ("Evaluation", "Budget", "Given")

STATKEY_POSITIVE_DECISIONS = ("Evaluation", "Decision Maker", "Positive Decisions")
STATKEY_DECISION_MAKER_BUDGET_THRESHOLD = ("Evaluation", "Decision Maker", "Budget Threshold")

STATKEY_MEAN_INST_QUALITY = ("Evaluation", "Acquisition", "Mean Instance Quality")
STATKEY_MEAN_QUALITY_GAIN = ("Evaluation", "Acquisition", "Mean Quality Gain")
STATKEY_ACQUISITIONS = ("Evaluation", "Acquisition", "Queries")
STATKEY_FEATURE_IMPORTANCES = ("Evaluation", "Acquisition", "Feature Importances")

STATKEY_IMPUTATIONS = ("Evaluation", "Imputation", "Imputed Features")
STATKEY_IMPUTATION_PERF = ("Evaluation", "Imputation", "Performance")

STATKEY_CHANGED_PREDICTION_AFTER_QUERY = ("Evaluation", "Queries", "Query Changed Prediction")
STATKEY_CORRECT_PREDICTION_AFTER_QUERY = ("Evaluation", "Queries", "Query Corrected Predictiion")
STATKEY_QUERIES_ASKED = ("Evaluation", "Queries", "Asked")
STATKEY_QUERIES_ANSWERED = ("Evaluation", "Queries", "Answered")

STATKEY_COMPUTATION_TIME = ("Evaluation", "Framework", "Computational Time")
