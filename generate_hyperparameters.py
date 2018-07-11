from sequences import *

def select_discrete(values, sample):
    K = len(values)
    diff = 1.0/K

    bound = 1-diff
    i = K-1

    while sample < bound:
        i -= 1
        bound -= diff

    assert i >= 0
    return values[i]


budget = 100   # I want to generate 100 vectors of hyperparameters
dimension = 2  # I have 17 hyperparameters
my_sampler = ScrHammersleySampler(dimension=dimension, budget=budget)   # I create my sampler.
sample = [my_sampler() for _ in range(budget)]      # I create my sample.

print(sample)

for s_bs, s_lr in sample:
    bs = select_discrete([100, 200, 300, 400, 500], s_bs)
    lr = select_discrete([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], s_lr)
    print(f"{bs}\t{lr}")