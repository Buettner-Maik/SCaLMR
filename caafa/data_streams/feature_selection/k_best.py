from .base import FeatureSelection

class KBest(FeatureSelection):
    def __init__(self, k):
        self.k = k

    def transform_one(self, x):
        return self._select_k_best(x, self.k)
        #return dict(sorted(x.items(), key=operator.itemgetter(1), reverse=True)[:self.k])

    def __str__(self):
        return f"{self.k}-Best"
    
    def __repr__(self):
        return f"{self.k}-Best"