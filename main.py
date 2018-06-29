import json
import sys
from collections import defaultdict
from queue import Queue

import nltk
import spacy
import torch
import torchtext
from torch.optim import Adam
from tqdm import tqdm

from model import SequentialChart, SnliModel

bs = 3
seqlen = 6
hd = 2



glove = torchtext.vocab.GloVe(name='6B', dim=100)


class Lexicon:
    def __init__(self):
        self.forward = {}
        self.backward = []

    def add(self, value):
        if value in self.forward:
            return self.forward[value]
        else:
            id = len(self.backward)
            self.forward[value] = id
            self.backward.append(value)
            return id

    # return None if does not exist
    def get_id(self, value):
        return self.forward.get(value)

    # return None if does not exist
    def get_value(self, id):
        if id >= len(self.backward):
            return None
        else:
            return self.backward[id]

    # add null entries until the next ID is next_id
    def pad(self, next_id):
        while len(self.backward) < next_id:
            self.add(f"***dummy{len(self.backward)}***")

    def __str__(self):
        return str(self.forward)

    def __repr__(self):
        return str(self)

def allowed(sentence_id, start, end):
    return True

def parse(sentence, sentence_index, maxlen, is_edge_allowed):
    n = len(sentence)
    operations = []
    edge_lex = Lexicon()

    # lexical productions
    for i in range(len(sentence)):
        item = (i,i+1)
        id = edge_lex.add(item)

    edge_lex.pad(maxlen)


    # build larger constituents, CKY-style
    for width in range(2, n+1):  # 2 <= width <= n
        for start in range(n-width+1):   # 0 <= start <= n-width
            # print(f"\nItem: {start}-{start+width}")
            if is_edge_allowed(sentence_index, start, start+width):
                ops = [] # decompositions for this item

                for split in range(1,width):  # 1 <= split <= width-1, width of left part
                    # print(f"Consider {start}-{start+split}-{start+width}")
                    left_item = (start, start+split)
                    right_item = (start+split, start+width)
                    left_id = edge_lex.get_id(left_item)
                    right_id = edge_lex.get_id(right_item)

                    if left_id is not None and right_id is not None:
                        # split exists
                        ops.append((left_id,right_id))
                        # print(f"Add op: #{left_id} to #{right_id}")

                if len(ops) > 0:
                    item = (start, start + width)
                    id = edge_lex.add(item)
                    operations.append(ops)
                    # print(f"Added ops for item {item} #{id}")

    return operations, edge_lex


def pad_list(list_to_pad, desired_length, filler):
    while len(list_to_pad) < desired_length:
        if type(filler) is list:
            x = filler.copy()
        else:
            x = filler

        list_to_pad.append(x)

def pad_operations(all_operations, max_ops_length):
    num_sentences = len(all_operations)

    # pad the ops sequence for each sentence to the max length of ops sequences
    for ops in all_operations:
        pad_list(ops, max_ops_length, [(0,0)])

    # pad the decompositions at each position to the same length
    for pos in range(max_ops_length):
        lengths = [len(all_operations[i][pos]) for i in range(num_sentences)]
        maxlen = max(lengths)

        for i in range(num_sentences):
            pad_list(all_operations[i][pos], maxlen, (0,0))


def convert_sentences(sentences, is_edge_allowed):
    # sentences: list(list(int)); len(sentences) = bs; len(sentences[0]) = length of first sentence; sentences[i][j] = id of j-th word in i-th sentence
    # is_edge_allowed: function, is_edge_allowed(i,j,k) = True iff edge from j-k in i is allowed
    # model: a SequentialChart object, which is used to map word ids to word embeddings

    max_sentence_length = max([len(sent) for sent in sentences])
    bs = len(sentences)
    parses = [parse(sent, sent_index, max_sentence_length, is_edge_allowed) for sent_index, sent in enumerate(sentences)]
    all_operations, all_edgelex = zip(*parses)

    num_decomps = [[len(decomps) for decomps in sent_operations] for sent_operations in all_operations]
    original_lengths = [len(sent_operations) for sent_operations in all_operations]
    max_ops_length = max(original_lengths)
    pad_operations(all_operations, max_ops_length)

    return all_operations, original_lengths, all_edgelex

import mmap

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


# TODO - I should bucket sentences so sentences of similar length are grouped together.
# TODO - Then only pad sentences in same bucket to same ops length.
# For now, the code is not wrong, will just be inefficient.

def tokenize(sentence):
    # NLTK - much much faster than Spacy
    return [token.lower() for token in nltk.word_tokenize(sentence)]

    # Spacy
    # tokens = tokenizer(sentence)
    # return [token.text.lower() for token in tokens]





# word_lex = Lexicon()

tokenizer = spacy.load('en_core_web_sm')

def word_lookup(words):
    def lookup(word):
        if word in glove.stoi:
            return glove.stoi[word]
        else:
            return -1

    return [lookup(word) for word in words]

# sentences = []
MAX_SENTENCES = 100
BATCHSIZE = 10

training_sent1 = []
training_sent2 = []
training_labels = []

max_sentence1_length = 0
max_sentence2_length = 0

snli_label_dict = {"neutral": 0, "contradiction":1, "entailment":2}

train_file = "/Users/akoller/Downloads/snli_1.0/snli_1.0_train.jsonl"
with open(train_file) as f:
    for line in tqdm(f, desc="Reading training sentences", total=get_num_lines(train_file)):
        j = json.loads(line)
        sentence1 = word_lookup(tokenize(j["sentence1"]))
        sentence2 = word_lookup(tokenize(j["sentence2"]))
        label = snli_label_dict[j["gold_label"]]

        training_sent1.append(sentence1)
        training_sent2.append(sentence2)
        training_labels.append(label)

        max_sentence1_length = max(max_sentence1_length, len(sentence1))
        max_sentence2_length = max(max_sentence2_length, len(sentence2))

        if len(training_labels) >= MAX_SENTENCES:
            break



# set up model and optimizer
model = SnliModel(10, 100, 10, glove)
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())





#
# sentences1 = [x[0] for x in training_data]
# sentences2 = [x[1] for x in training_data]
#
# print("convert")
# ops1, oopl1, edgelex1 = convert_sentences(sentences1, allowed)
# ops2, oopl2, edgelex2 = convert_sentences(sentences2, allowed)
# print("forward")
# print(model(sentences1, ops1, oopl1, sentences2, ops2, oopl2))
# sys.exit(0)



for epoch in range(10):
    optimizer.zero_grad()
    total_loss = 0

    for batch in range(int(MAX_SENTENCES/BATCHSIZE)):
        print(f"\nbatch {batch}")
        s1 = training_sent1[batch*BATCHSIZE : (batch+1)*BATCHSIZE]
        s2 = training_sent2[batch*BATCHSIZE : (batch+1)*BATCHSIZE]
        lab = torch.LongTensor(training_labels[batch*BATCHSIZE : (batch+1)*BATCHSIZE])

        print("convert")
        # TODO - I think this can happen once in the beginning
        ops1, oopl1, edgelex1 = convert_sentences(s1, allowed)
        ops2, oopl2, edgelex2 = convert_sentences(s2, allowed)
        print(f"max oopl1: {max(oopl1)}")
        print(f"max oopl2: {max(oopl2)}")

        print("forward")
        predictions = model(s1, ops1, oopl1, s2, ops2, oopl2)

        loss = criterion(predictions, lab)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        print(f"loss in batch {batch}: {loss.item()}")

    print(total_loss)






#
#
# all_operations, original_opseq_lengths, all_edgelex = convert_sentences(sentences, allowed, model)
# max_sentence_length = max([len(sent) for sent in sentences])
#
# for i, sentence in enumerate(sentences):
#     print(f"{i+1}: {sentence}")
#
# # print(f"msl: {max_sentence_length}; ool: {original_opseq_lengths}")
#
# chart = model.chart_for_batch(sentences, glove, original_opseq_lengths, max_sentence_length)
#
# result = model(chart, all_operations, max_sentence_length, original_opseq_lengths)
# print(result)
# #def forward(self, chart, operations, start_index, original_operations_lengths):
#
#
# print(chart.size())
# print(chart[0,0,:])
# sys.exit(0)
