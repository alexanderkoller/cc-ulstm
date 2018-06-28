import torch
import torch.nn.functional as F
from torch.nn import Module


class SequentialChart(Module):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        # self.input_dim = input_dim

        # self.W = torch.FloatTensor((5 * self.hidden_dim, self.input_dim), requires_grad=True)
        self.U = torch.zeros((5 * self.hidden_dim, 2 * self.hidden_dim), requires_grad=True)   # TODO - random initialization
        self.b = torch.zeros((5 * self.hidden_dim, ), requires_grad=True)
        self.u = 0 # TODO

    def forward(self, chart, operations, start_index):
        # chart: (bs, seqlen, 2*hd)           chart[i,j,:] = (c,h) for position #j of sentence #i in batch
        # with chart[0,:,:] all zero; so zero premises can be used for nop

        # operations: [[(L1,R1),...,(Lb,Rb)], ..., [(L1,R1),...,(Lb,Rb)]]    # b = batchsize; Li, Ri are indices into chart dimension 1
        # NOT: operations: [(bs,2), ..., (bs,2)]   operations[i][j,:] = (left premise index, right premise index) for step #i on sentence #j in batch

        bs = chart.size()[0]

        for step, ops in enumerate(operations):
            # ops : [(L1,R1), ..., (Lb,Rb)]
            # NOT: ops[j,:] = (L index, R index) for sentence #j   # TODO -> expand to multiple premises, op will gain extra dimension

            # TODO - this is probably expensive. See if they can be made faster.
            chart_entries = [torch.stack((chart[b,l,:], chart[b,r,:])) for b, (l,r) in enumerate(ops)]   # len=bs; chart_entries[b]: (2, 2*hd)
            chart_entries = torch.stack(chart_entries) # (bs, 2, 2*hd)

            cc = chart_entries[:,:,:self.hidden_dim]   # (bs, 2, hd)
            ccL = cc[:,0,:].squeeze(dim=1) # (bs, hd)
            ccR = cc[:,1,:].squeeze(dim=1) # (bs, hd)
            hh = chart_entries[:,:,self.hidden_dim:].contiguous().view(-1,2*self.hidden_dim,1)   # (bs, 2*hd, 1)  # TODO - contiguous expensive?

            print(f"U: {self.U.size()}")
            print(f"cc: {cc.size()}")
            print(f"b: {self.b.size()}")

            UE = self.U.expand(bs, -1, -1)  # (bs, 5*hd, 2*hd)
            print(f"UE: {UE.size()}")


            # batched evaluation of TreeLSTM cell
            mult = torch.bmm(UE, hh).squeeze(dim=2)
            preact = mult + self.b    # (bs, 5*hd)
            i = F.sigmoid(preact[:,:self.hidden_dim]) # (bs, hd)
            fL = F.sigmoid(preact[:, self.hidden_dim:2*self.hidden_dim]) # (bs,hd)
            fR = F.sigmoid(preact[:, 2*self.hidden_dim:3*self.hidden_dim]) # (bs,hd)
            o = F.sigmoid(preact[:, 3*self.hidden_dim:4*self.hidden_dim]) # (bs,hd)
            u = F.tanh(preact[:, 4*self.hidden_dim:5*self.hidden_dim]) # (bs,hd)
            c = fL*ccL + fR*ccR + i*u   # (bs,hd)
            h = o * F.tanh(c)  # (bs,hd)
            print(f"c: {c.size()}")

            # update chart
            chart[:, start_index+step, :self.hidden_dim] = c
            chart[:, start_index+step, self.hidden_dim:] = h

        return chart


