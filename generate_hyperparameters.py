import sys

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
dimension = 3  # I have 17 hyperparameters
my_sampler = ScrHammersleySampler(dimension=dimension, budget=budget)   # I create my sampler.
sample = [my_sampler() for _ in range(budget)]      # I create my sample.

for s_bs, s_lr, s_hd in sample:
    # batchsize
    bs = select_discrete([100, 200, 300, 400, 500], s_bs)

    # learning rate
    lr = select_discrete([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], s_lr)

    # hidden_dim
    hd = select_discrete([50,100,150,200,250,300,1000], s_hd)

    print(f"{bs}\t{lr}\t{hd}")


