from sortedcontainers import SortedList
import collections
import math
from .base import DecisionMaker

class IncrementalPercentileFilter(DecisionMaker):
    """Incrementally adjusts a threshold to decide positive or negative actions
    """
    def __init__(self, budget_threshold, window_size):
        self.counter = 0
        self.window_size = window_size
        self.budget_threshold = budget_threshold
        
        self.values_queue = collections.deque()
        self.values = SortedList()

    def learn_one(self, x):
        self.counter += 1

        i = (self.counter - 1) % self.window_size

        if self.counter > self.window_size:
            oldest_val = self.values_queue.popleft()
            self.values.remove(oldest_val)
        
        self.values_queue.append(x)
        self.values.add(x)

    def transform_one(self, x):
        return self.values[math.floor(min(self.window_size, self.counter) * (1 - self.budget_threshold))] <= x
    
    def __str__(self):
        return "IncrementalPercentileFilter"
    
    def __repr__(self):
        return "IncrementalPercentileFilter"