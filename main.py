
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

for param in model.parameters():
    print(type(param.data), param.size())


operations = [[[(0,1)], [(0,1)], [(0,1)]], [[(1,2)], [(1,2)], [(1,2)]], [[(0,4),(3,2)], [(0,4),(0,0)], [(0,4),(0,0)]]] # TODO - dealing with diff ambiguity?

result = model(chart, operations, 3)

print(result)
# print(result.size())

