
## https://github.com/fairinternal/dfoptim/blob/master/sequences.py
## Olivier Teytoud

# Samplers in [0,1]^d.

import numpy as np

SAMPLERS = []


def register_sampler(cls):
    global SAMPLERS
    SAMPLERS += [cls]
    return cls


class Sampler(object):

    def __init__(self, dimension, budget=None):
        self.dimension = dimension
        self.budget = budget
        self.index = 0

    def internal_sampler(self):
        raise NotImplementedError("Missing sampling function! which is quite necessary for a sampler.")

    def __call__(self):
        sample = self.internal_sampler()
        self.index += 1
        return sample


@register_sampler
class RandomSampler(Sampler):

    def internal_sampler(self):
        return np.random.uniform(0, 1, self.dimension)


# TODO: add a seed dependency
@register_sampler
class HaltonSampler(Sampler):

    def __init__(self, dimension, budget=None, scrambling=False):
        super(HaltonSampler, self).__init__(dimension, budget)
        self.primes = [2]
        current_candidate = 3
        while len(self.primes) < dimension:
            while any([(current_candidate % p) == 0 for p in self.primes]):
                current_candidate += 2
            self.primes += [current_candidate]
        if scrambling == True:
            np.random.seed()
            self.permutations = []
            for p in self.primes:
                self.permutations += [[0] + list(np.random.choice(range(1, p), p - 1, replace=False))]
        else:
            self.permutations = [range(p) for p in self.primes]

    def vdc(self, n, base=2, permut=None):
        vdc, denom = 0, 1
        n += 1
        while n:
            denom *= base
            n, remainder = divmod(n, base)
            remainder = permut[remainder]
            vdc += float(remainder) / float(denom)
        return vdc

    def internal_sampler(self):
        sample = [self.vdc(self.index, p, sigma) for p, sigma in zip(self.primes, self.permutations)]
        return sample


@register_sampler
class ScrHaltonSampler(HaltonSampler):

    def __init__(self, dimension, budget=None):
        super(ScrHaltonSampler, self).__init__(dimension, budget, scrambling=True)


@register_sampler
class HammersleySampler(HaltonSampler):

    def __init__(self, dimension, budget, scrambling=False):
        assert budget is not None
        super(HammersleySampler, self).__init__(dimension - 1, budget, scrambling)

    def internal_sampler(self):
        return np.asarray(
            [(self.index + .5) / float(self.budget)] + list(super(HammersleySampler, self).internal_sampler()))


@register_sampler
class ScrHammersleySampler(HammersleySampler):

    def __init__(self, dimension, budget):
        super(ScrHammersleySampler, self).__init__(dimension, budget, scrambling=True)


def draw_sampler(sampling_class, budget, dimension):
    sampler = sampling_class(dimension, budget=budget)
    sample = [sampler() for _ in xrange(budget)]
    for i in xrange(dimension):
        for j in xrange(i + 1, dimension):
            print
            "plotting coordinates " + str(i) + "," + str(j)
            tab = [["." for _ in xrange(80)] for _ in xrange(20)]
            for s in sample:
                x = int(s[i] * 20)
                y = int(s[j] * 80)
                tab[x][y] = "*"
            for t in tab:
                print
                "".join(t)


def rescale(sample):
    sample_min = [min([s[i] for s in sample]) for i in xrange(len(sample[0]))]
    sample_max = [max([s[i] for s in sample]) for i in xrange(len(sample[0]))]
    epsilon = min(sample_min + [1 - s for s in sample_max] + [1e-15])
    assert epsilon > 0., str(epsilon) + str(sample_min) + str(sample_max)
    factor = [(1 - 2 * epsilon) / (ma - mi) for ma, mi in zip(sample_max, sample_min)]
    return [[epsilon + f * (s - m) for f, s, m in zip(factor, guy, sample_min)] for guy in sample]

# draw_sampler(ScrHammersleySampler, 30, 2)
#
#
# for cls in SAMPLERS:
#  print str(cls) + " has type " + str(type(cls))
#  draw_sampler(cls, 30, 2)

