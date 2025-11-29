# TODO: Make proper
from .smr import _SupervisedMeritRankingOptions

class NoAcquisition():
    def __init__(self):
        self.feature_importance = {}
        self.feature_selection = None
        self.options = _SupervisedMeritRankingOptions()
    
    def learn_one(self, x, y):
        pass

    def get_quality(self, x, y, y_conf):
        return 0.0
    
    def get_set(self, x, y, y_conf):
        return {}