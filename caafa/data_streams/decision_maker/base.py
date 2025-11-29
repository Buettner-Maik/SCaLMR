class DecisionMaker():
    def learn_one(self, x):
        pass

    def transform_one(self, x):
        pass

class AlwaysYesDecisionMaker(DecisionMaker):
    def transform_one(self, x):
        return True
    
    def __str__(self):
        return "AlwaysYes"
    
    def __repr__(self):
        return "AlwaysYes"

class AlwaysNoDecisionMaker(DecisionMaker):
    def transform_one(self, x):
        return False
    
    def __str__(self):
        return "AlwaysNo"
    
    def __repr__(self):
        return "AlwaysNo"

class SimpleDecisionMaker(DecisionMaker):
    """Tracks its own budget which is added per learn_one call
    and subtracted per transform_one call
    """
    def __init__(self, b_gain=1.0, d_cost=1.0, b_initial=0.0):
        """
        :param b_gain: The budget to add per learn_one call
        :param d_cost: The cost to compare against and to subtract per transform_one call
        :param b_initial: The initial budget
        """
        self.b_gain = b_gain
        self.d_cost = d_cost
        self.b_current = b_initial

    def learn_one(self, x):
        self.b_current += self.b_gain

    def transform_one(self, x):
        if self.b_current > self.d_cost:
            self.b_current -= self.d_cost
            return True
        return False
    
    def __str__(self):
        return "SimpleDecisionMaker"
    
    def __repr__(self):
        return "SimpleDecisionMaker"