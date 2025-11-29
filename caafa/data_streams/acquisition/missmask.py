import random

class MissMask():
    """Methods about generating a mask of missing values
    """
    def __init__(self, mask_generator):
        """mask_generator will be called as a generator to return dicts of type {"key":bool}
        """
        self.mask_generator = mask_generator

    def __next__(self):
        return self.mask_generator.__next__()

    def __iter__(self):
        return self

    def from_uniform_rngs(miss_seeds, miss_chances, skip):
        """Makes a stream like mask of misses
        """
        miss_rngs = {f:random.Random(miss_seed) for f, miss_seed in miss_seeds.items()}

        def gen_random(rngs, thrs, skip):
            i = 0
            while True:
                i += 1
                if i <= skip:
                    yield {f:False for f in rngs}
                else:
                    yield {f:rng.random() < thrs[f] for f, rng in rngs.items()}

        mask = gen_random(miss_rngs, miss_chances, skip)
        
        return MissMask(mask_generator=mask)
