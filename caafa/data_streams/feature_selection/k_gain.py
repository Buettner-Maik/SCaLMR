from .base import FeatureSelection

class KGreedyGain(FeatureSelection):
    def __init__(self, k, all_feature_names, quality_function):
        self.k = k
        self.features = all_feature_names
        self.quality_function = quality_function

    def transform_one(self, x):
        current_set = {f:1.0 for f in self.features
                       if f not in x}
        set_size = 0
        current_quality = self.quality_function(current_set)
        
        remaining_set = x.copy()
        x = {}

        while set_size < self.k and any(remaining_set):
            
            qualities = {f:self.quality_function(current_set | {f:remaining_set[f]}) 
                         for f in remaining_set}
            max_f = max(qualities, key=qualities.get)
            max_quality = qualities[max_f]

            if current_quality > max_quality:
                break

            current_set[max_f] = remaining_set[max_f]
            x[max_f] = remaining_set[max_f]
            del remaining_set[max_f]

            current_quality = max_quality
            set_size += 1

        return x
        
    def __str__(self):
        return f"{self.k}-GreedyGain"
    
    def __repr__(self):
        return f"{self.k}-GreedyGain"