from river.preprocessing import PreviousImputer

class PreviousImputerWrapper(PreviousImputer):
    """Rivers implementation of the PreviousImputer does not make a copy
    before transform. This just wraps a copy around it.
    """
    def transform_one(self, x):
        x = x.copy()
        return super().transform_one(x)