
import torch

from model import SequentialChart

bs = 3
seqlen = 6
hd = 2

chart = torch.zeros(bs, seqlen, 2*hd)
chart[:,0,:] = torch.rand(bs, 2*hd)
chart[:,1,:] = torch.rand(bs, 2*hd)
chart[:,2,:] = torch.rand(bs, 2*hd)
# print(chart)

model = SequentialChart(hd)


operations = [[(0,1), (0,1), (0,1)], [(1,2), (1,2), (1,2)], [(0,4), (0,4), (0,4)]]

result = model.forward(chart, operations, 3)
print(result)
print(result.size())

