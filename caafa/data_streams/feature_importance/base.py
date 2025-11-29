# TODO: Make proper

class FeatureImportance():

    def update(self, x, y, **kwargs):
        pass

    def revert(self, x, y, **kwargs):
        pass

    def get(self):
        pass

    def __str__(self):
        return str(self.get())
    
    def __repr__(self):
        return str(self.get())

class FixedImportance(FeatureImportance):
    def __init__(self, value):
        self.value = value
    
    def get(self):
        return self.value

class SumImportance(FeatureImportance):
    def __init__(self, feature_importances):
        self.feature_importances = feature_importances
    
    def get(self):
        return sum([fi.get() for fi in self.feature_importances.values()])