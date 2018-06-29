import time

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, Linear


class SnliModel(Module):
    def __init__(self, hidden_dim, embedding_dim, mlp_dim, glove):
        super(SnliModel, self).__init__()

        self.sentence_model = SequentialChart(hidden_dim, embedding_dim, glove)
        self.clayer1 = Linear(2*hidden_dim, mlp_dim)
        self.clayer2 = Linear(mlp_dim, 3)

    def forward(self, sentences1, operations1, oopl1, sentences2, operations2, oopl2):
        s1 = self.sentence_model(sentences1, operations1, oopl1) # (bs, hd)
        s2 = self.sentence_model(sentences2, operations2, oopl2) # (bs, hd)
        conc = torch.cat((s1, s2), dim=1) # (bs, 2*hd)

        internal1 = F.relu(self.clayer1(conc)) # (bs, mlp_dim)
        internal2 = F.softmax(self.clayer2(internal1), dim=1) # (bs, 3)

        return internal2

class SequentialChart(Module):
    def __init__(self, hidden_dim, embedding_dim, glove):
        super(SequentialChart, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.glove = glove

        self.W = Parameter(torch.zeros((5 * self.hidden_dim, self.embedding_dim), requires_grad=True))
        self.U = Parameter(torch.zeros((5 * self.hidden_dim, 2 * self.hidden_dim), requires_grad=True))
        self.b = Parameter(torch.zeros((5 * self.hidden_dim, ), requires_grad=True))
        self.energy_u = Parameter(0.01*torch.ones((self.hidden_dim,), requires_grad=True))

        self.unk_embedding = Parameter(torch.zeros((self.embedding_dim,), requires_grad=True))

        for p in (self.W, self.U, self.energy_u, self.unk_embedding):
            torch.nn.init.uniform_(p)

        self.temperature = torch.tensor([1.])

    def ch(self, preact, ccL, ccR):
        i = F.sigmoid(preact[:, :, :self.hidden_dim])  # (bs, amb, hd)
        fL = F.sigmoid(preact[:, :, self.hidden_dim:2 * self.hidden_dim])  # (bs,amb,hd)
        fR = F.sigmoid(preact[:, :, 2 * self.hidden_dim:3 * self.hidden_dim])  # (bs,amb,hd)
        o = F.sigmoid(preact[:, :, 3 * self.hidden_dim:4 * self.hidden_dim])  # (bs,amb,hd)
        u = F.tanh(preact[:, :, 4 * self.hidden_dim:5 * self.hidden_dim])  # (bs,amb,hd)

        c = fL * ccL + fR * ccR + i * u  # (bs,amb,hd)
        h = o * F.tanh(c)  # (bs,amb,hd)
        return c,h

    def chart_for_batch(self, sentences, word_embeddings, original_opseq_lengths, max_sentence_length):
        bs = len(sentences)
        chart = torch.zeros(bs, max_sentence_length + max(original_opseq_lengths), 2 * self.hidden_dim)

        for i in range(max_sentence_length):
            preact = torch.zeros(bs, 1, 5 * self.hidden_dim)
            for b, sentence in enumerate(sentences):
                if i >= len(sentence):
                    # no word at this position
                    preact[b] = torch.zeros((1,5*self.hidden_dim))
                else:
                    wordid = sentence[i]
                    emb = word_embeddings.vectors[wordid] if wordid >= 0 else self.unk_embedding  # (embdim)
                    preact[b] = self.W @ emb + self.b  # (5*hd), broadcast into (1, 5*hd)

            c,h = self.ch(preact, 0, 0) # 2x (bs, 1, hd)
            chart[:, i, :self.hidden_dim] = c.squeeze(dim=1)
            chart[:, i, self.hidden_dim:] = h.squeeze(dim=1)

        return chart


    def forward(self, sentences, operations, original_operations_lengths):
        # chart: (bs, seqlen, 2*hd)           chart[i,j,:] = (c,h) for position #j of sentence #i in batch
        # with chart[0,:,:] all zero; so zero premises can be used for nop  ## XXX this is no longer true

        # operations = list(opseq), where len(operations) = #sentences in batch, and opseq is list of operations for this sentence
        # opseq = list(op), where len(opseq) = #parsing steps for this sentence
        # ops = [(L,R), ..., (L,R)], i.e. all ways in which this item can be built from a left and right part; L,R are indices into chart rows (dimension 1)

        # operations and opseq have been padded to max length. Original lengths of the opseqs (before padding) are in original_operations_lengths.
        device = self.W.get_device()

        max_sentence_length = max(len(sentence) for sentence in sentences)
        chart = self.chart_for_batch(sentences, self.glove, original_operations_lengths, max_sentence_length).to(device)
        start_index = max_sentence_length

        bs = chart.size()[0]
        assert bs == len(operations)

        max_opseq_length = max([len(opseq) for opseq in operations])  # max length of the opseq's

        total_setup_time = 0
        total_forward_time = 0

        for step in range(max_opseq_length):
            # TODO - this is probably expensive. See if it can be made faster.
            start = time.time()
            chart_entries = [
                torch.stack(
                    [torch.stack((chart[b,l,:], chart[b,r,:])) for l,r in operations[b][step]] # list of amb entries of shape (2, 2*hd)
                ) # (amb, 2, 2*hd)
                for b in range(bs)]   # len=bs; chart_entries[b]: (2, 2*hd)

            chart_entries = torch.stack(chart_entries) # (bs, amb, 2, 2*hd)
            amb = chart_entries.size()[1] # #decompositions of items in this step

            mid = time.time()

            cc = chart_entries[:,:,:,:self.hidden_dim]   # (bs, amb, 2, hd)
            ccL = cc[:,:,0,:].squeeze(dim=2) # (bs, amb, hd)
            ccR = cc[:,:,1,:].squeeze(dim=2) # (bs, amb, hd)
            hh = chart_entries[:,:,:,self.hidden_dim:].contiguous().view(-1,2*self.hidden_dim,1)   # (bs*amb, 2*hd, 1)  # TODO - contiguous expensive?

            # batched evaluation of TreeLSTM cell
            UE = self.U.expand(bs*amb, -1, -1)  # (bs*amb, 5*hd, 2*hd)
            mult = torch.bmm(UE, hh)        # (bs*amb, 5*hd, 1)
            mult = mult.view(bs, amb, 5*self.hidden_dim) # (bs, amb, 5*hd)

            preact = mult + self.b    # (bs, amb, 5*hd)
            c, h = self.ch(preact, ccL, ccR)

            # combine decompositions of this item
            expanded_u = self.energy_u.expand((bs, amb, -1)) # (bs, amb, hd)
            e = F.cosine_similarity(expanded_u, h, dim=2)    # (bs, amb)

            s = F.softmax(e/self.temperature[0], dim=1).view(bs,amb,1)      # (bs, amb, 1)
            combined_c = (s*c).sum(dim=1) # (bs, hd)
            combined_h = (s*h).sum(dim=1) # (bs, hd)

            # update chart
            chart[:, start_index+step, :self.hidden_dim] = combined_c
            chart[:, start_index+step, self.hidden_dim:] = combined_h

            end = time.time()
            total_setup_time += (mid-start)
            total_forward_time += (end-mid)


        print(f"model: setup={total_setup_time}s, forward={total_forward_time}s")

        # return a tensor with h for the final state of each sentence in the minibatch
        ret = torch.stack([chart[b, original_operations_lengths[b]-1, self.hidden_dim:] for b in range(bs)]) # (bs, hd)
        return ret



