# TODO: Make proper
import operator

class FeatureSelection():
    def _select_k_best(self, x, k):
        return dict(sorted(x.items(), key=operator.itemgetter(1), reverse=True)[:k])

    def learn_one(self, x):
        pass

    def transform_one(self, x):
        pass
    
    def forget_one(self, x):
        pass

class AlwaysSelect(FeatureSelection):
    def __init__(self, chosen):
        self.chosen = chosen

    def transform_one(self, x):
        return {key:item for key, item in x.items() if key in self.chosen}

    def __str__(self):
        return f"Always-{self.chosen}"
    
    def __repr__(self):
        return f"Always-{self.chosen}"