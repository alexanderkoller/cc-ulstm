import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter


class SequentialChart(Module):
    def __init__(self, hidden_dim):
        super(SequentialChart, self).__init__()

        self.hidden_dim = hidden_dim

        self.U = Parameter(torch.zeros((5 * self.hidden_dim, 2 * self.hidden_dim), requires_grad=True))
        self.b = Parameter(torch.zeros((5 * self.hidden_dim, ), requires_grad=True))
        self.energy_u = Parameter(0.01*torch.ones((self.hidden_dim,), requires_grad=True))

        for p in (self.U, self.energy_u):
            torch.nn.init.uniform_(p)

        self.temperature = torch.tensor([1.])

    def forward(self, chart, operations, start_index):
        # chart: (bs, seqlen, 2*hd)           chart[i,j,:] = (c,h) for position #j of sentence #i in batch
        # with chart[0,:,:] all zero; so zero premises can be used for nop

        # operations: [[[(L1,R1), (L1',R1') ...],...,[(Lb,Rb), (Lb',Rb') ...]], ..., ]    # b = batchsize; Li, Ri are indices into chart dimension 1

        bs = chart.size()[0]

        for step, ops in enumerate(operations):
            # ops : [[(L1,R1), (L1',R1') ...], ..., [(Lb,Rb), (Lb',Rb') ...]]
            # ops[b] is a list of decompositions (L,R) for the chart entry #step+start_index for sentence #b in batch

            # TODO - this is probably expensive. See if they can be made faster.
            chart_entries = [
                torch.stack(
                    [torch.stack((chart[b,l,:], chart[b,r,:])) for l,r in decomps] # list of amb entries of shape (2, 2*hd)
                ) # (amb, 2, 2*hd)
                for b, decomps in enumerate(ops)]   # len=bs; chart_entries[b]: (2, 2*hd)
            chart_entries = torch.stack(chart_entries) # (bs, amb, 2, 2*hd)

            # print(f"ce: {chart_entries.size()}")
            amb = chart_entries.size()[1] # #decompositions of items in this step

            cc = chart_entries[:,:,:,:self.hidden_dim]   # (bs, amb, 2, hd)
            ccL = cc[:,:,0,:].squeeze(dim=2) # (bs, amb, hd)
            ccR = cc[:,:,1,:].squeeze(dim=2) # (bs, amb, hd)
            hh = chart_entries[:,:,:,self.hidden_dim:].contiguous().view(-1,2*self.hidden_dim,1)   # (bs*amb, 2*hd, 1)  # TODO - contiguous expensive?

            # print(f"ccL: {ccL.size()}")
            # print(f"U: {self.U.size()}")
            # print(f"hh: {hh.size()}")
            # print(f"b: {self.b.size()}")

            UE = self.U.expand(bs*amb, -1, -1)  # (bs*amb, 5*hd, 2*hd)
            # print(f"UE: {UE.size()}")


            # batched evaluation of TreeLSTM cell
            mult = torch.bmm(UE, hh)        # (bs*amb, 5*hd, 1)
            mult = mult.view(bs, amb, 5*self.hidden_dim) # (bs, amb, 5*hd)
            # print(f"mult: {mult.size()}")

            preact = mult + self.b    # (bs, amb, 5*hd)
            # print(f"preact: {preact.size()}")

            i = F.sigmoid(preact[:, :, :self.hidden_dim]) # (bs, amb, hd)
            fL = F.sigmoid(preact[:, :, self.hidden_dim:2*self.hidden_dim]) # (bs,amb,hd)
            fR = F.sigmoid(preact[:, :, 2*self.hidden_dim:3*self.hidden_dim]) # (bs,amb,hd)
            o = F.sigmoid(preact[:, :, 3*self.hidden_dim:4*self.hidden_dim]) # (bs,amb,hd)
            u = F.tanh(preact[:, :, 4*self.hidden_dim:5*self.hidden_dim]) # (bs,amb,hd)

            c = fL*ccL + fR*ccR + i*u   # (bs,amb,hd)
            h = o * F.tanh(c)  # (bs,amb,hd)

            # print(f"c: {c.size()}")
            # print(f"h: {h.size()}")

            # combine decompositions of this item
            expanded_u = self.energy_u.expand((bs, amb, -1)) # (bs, amb, hd)
            e = F.cosine_similarity(expanded_u, h, dim=2)    # (bs, amb)
            # print(f"e: {e.size()}")

            s = F.softmax(e/self.temperature[0], dim=1).view(bs,amb,1)      # (bs, amb, 1)
            combined_c = (s*c).sum(dim=1) # (bs, hd)
            combined_h = (s*h).sum(dim=1) # (bs, hd)

            # update chart
            chart[:, start_index+step, :self.hidden_dim] = combined_c
            chart[:, start_index+step, self.hidden_dim:] = combined_h

        return chart


