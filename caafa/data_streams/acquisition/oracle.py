import random
from sortedcontainers import SortedDict

class OracleMultiReferenceRememberer():
    def __init__(self, random_seed, delay_min=0, delay_max=0):
        """Stores queries and generates delays if specified

        delay_min, delay_max are inclusive random min and max values for the uniform random sampler
        """
        self._seed = random_seed
        self.rng = random.Random(self._seed)
        self.delay_min = delay_min
        self.delay_max = delay_max

        self.queries = SortedDict()
        self.old_insts = {}
        self.old_inst_references = {}

    def _get_delay(self):
        return self.rng.randint(self.delay_min, self.delay_max)

    def query_one(self, index, key, true_value, old_inst):
        available_at = self._get_delay() + index
        
        if index not in self.old_insts:
            self.old_insts[index] = old_inst.copy()
            self.old_inst_references[index] = 0
        self.old_inst_references[index] += 1

        if available_at not in self.queries:
            self.queries[available_at] = SortedDict()
        self.queries[available_at][(index, key)] = (true_value, self.old_insts[index])

    def retrieve_many(self, index):
        """return {(index, key):(true_value, old_value)}"""
        answered_queries = self.queries.pop(index, default={})
        for index, feature_key in answered_queries:
            self.old_inst_references[index] -= 1
            if self.old_inst_references[index] == 0:
                del self.old_inst_references[index]
                del self.old_insts[index]
        
        return answered_queries

class OracleValueRememberer():
    def __init__(self, random_seed, delay_min=0, delay_max=0):
        """Stores queries and generates delays if specified

        delay_min, delay_max are inclusive random min and max values for the uniform random sampler
        """
        self._seed = random_seed
        self.rng = random.Random(self._seed)
        self.delay_min = delay_min
        self.delay_max = delay_max

        self.queries = SortedDict()

    def _get_delay(self):
        return self.rng.randint(self.delay_min, self.delay_max)

    def query_one(self, index, key, true_value, old_value):
        available_at = self._get_delay() + index

        if available_at not in self.queries:
            self.queries[available_at] = SortedDict()
        self.queries[available_at][(index, key)] = (true_value, old_value)

    def retrieve_many(self, index):
        """return {(index, key):(true_value, old_value)}"""
        return self.queries.pop(index, default={})