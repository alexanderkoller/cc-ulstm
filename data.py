import nltk
import json
from collections import namedtuple

import torchtext
from tqdm import tqdm

from chart_constraints import AllAllowedChartConstraints, BeginEndChartConstraints
from util import get_num_lines


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

def parse(sentence, sentence_index, maxlen, cc, sentence_index_offset, original_index):
    global num_edges_considered
    global num_edges_allowed

    n = len(sentence)
    operations = []
    edge_lex = Lexicon()

    # lexical productions
    for i in range(len(sentence)):
        item = (i, i + 1)
        id = edge_lex.add(item)

    edge_lex.pad(maxlen)

    # build larger constituents, CKY-style
    for width in range(2, n):  # 2 <= width <= n
        for start in range(n - width):  # 0 <= start <= n-width
            # print(f"\nItem: {start}-{start+width} of {n}")
            num_edges_considered += 1

            original_sentence_index = original_index[sentence_index_offset + sentence_index]

            if cc.is_edge_allowed(original_sentence_index, start, start + width):
                num_edges_allowed += 1
                ops = []  # decompositions for this item

                for split in range(1, width):  # 1 <= split <= width-1, width of left part
                    # print(f"Consider {start}-{start+split}-{start+width}")
                    left_item = (start, start + split)
                    right_item = (start + split, start + width)
                    left_id = edge_lex.get_id(left_item)
                    right_id = edge_lex.get_id(right_item)

                    if left_id is not None and right_id is not None:
                        # split exists
                        ops.append((left_id, right_id))
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
        pad_list(ops, max_ops_length, [(0, 0)])

    # pad the decompositions at each position to the same length
    for pos in range(max_ops_length):
        lengths = [len(all_operations[i][pos]) for i in range(num_sentences)]
        maxlen = max(lengths)

        for i in range(num_sentences):
            pad_list(all_operations[i][pos], maxlen, (0, 0))

def convert_sentences(sentences, cc, sentence_index_offset, original_index):
    # sentences: list(list(int)); len(sentences) = bs; len(sentences[0]) = length of first sentence; sentences[i][j] = id of j-th word in i-th sentence
    # cc: object with chart constraints; cc.is_edge_allowed(i,j,k) = True iff edge from j-k in i is allowed
    # sentence_index_offset: offset of first sentence in this batch in the global list of sentences (for determining chart constraints)
    # original_index: list that specifies the original sentence index for each sentence

    max_sentence_length = max([len(sent) for sent in sentences])
    bs = len(sentences)
    parses = [parse(sent, sent_index, max_sentence_length, cc, sentence_index_offset, original_index) for
              sent_index, sent in enumerate(sentences)]
    all_operations, all_edgelex = zip(*parses)

    num_decomps = [[len(decomps) for decomps in sent_operations] for sent_operations in all_operations]
    original_lengths = [len(sent_operations) for sent_operations in all_operations]
    max_ops_length = max(original_lengths)
    pad_operations(all_operations, max_ops_length)

    return all_operations, original_lengths, all_edgelex

# TODO - I should bucket sentences so sentences of similar length are grouped together.
# TODO - Then only pad sentences in same bucket to same ops length.
# For now, the code is not wrong, will just be inefficient.

def tokenize(sentence):
    # NLTK - much much faster than Spacy
    return [token.lower() for token in nltk.word_tokenize(sentence)]

    # Spacy
    # tokens = tokenizer(sentence)
    # return [token.text.lower() for token in tokens]

def word_lookup(words, glove):
    def lookup(word):
        if word in glove.stoi:
            return glove.stoi[word]
        else:
            return -1

    return [lookup(word) for word in words]


#     train_file = "data/snli_1.0/snli_1.0_train.jsonl"

def get_data(train_file, batchsize, limit, sort, cc):
    glove = torchtext.vocab.GloVe(name='6B', dim=100)

    global num_edges_considered
    global num_edges_allowed

    num_edges_considered = 0
    num_edges_allowed = 0


    MAX_SENTENCES = limit

    training_sent1 = []
    training_sent2 = []
    training_labels = []

    max_sentence1_length = 0
    max_sentence2_length = 0

    snli_label_dict = {"neutral": 0, "contradiction": 1, "entailment": 2, "-": 0}  # TODO - I don't understand the "-" label, double-check it

    with open(train_file) as f:
        for line in tqdm(f, desc="Reading training sentences", total=get_num_lines(train_file)):
            j = json.loads(line)
            sentence1 = word_lookup(tokenize(j["sentence1"]), glove)
            sentence2 = word_lookup(tokenize(j["sentence2"]), glove)
            label = snli_label_dict[j["gold_label"]]

            training_sent1.append(sentence1)
            training_sent2.append(sentence2)
            training_labels.append(label)

            max_sentence1_length = max(max_sentence1_length, len(sentence1))
            max_sentence2_length = max(max_sentence2_length, len(sentence2))

            if len(training_labels) >= MAX_SENTENCES:
                break

    # if requested, sort inputs by length of sent1
    if sort:
        print("Sorting sentences by length ...")
        # print(f"before: {[len(x) for x in training_sent1[:100]]}")
        to_sort = zip(training_sent1, training_sent2, training_labels, range(len(training_sent1)))
        srted = sorted(to_sort, key=lambda x: len(x[0]))  # TODO - try len(x[0])+len(x[1])
        training_sent1, training_sent2, training_labels, original_index = zip(*srted)
        # print(f"after: {[len(x) for x in training_sent1[:100]]}")
        print("Done.")
    else:
        original_index = list(range(len(training_sent1)))

    # parse all the sentences once
    # (may need to do this for each batch, if it doesn't fit in memory)
    ParsingResult = namedtuple("ParsingResult", ["sentences", "ops", "oopl", "edgelex"])
    batched_parses = []

    if cc is None:
        cc_sent1 = AllAllowedChartConstraints()
        cc_sent2 = AllAllowedChartConstraints()
    else:
        print("Reading chart constraints for sent1 ...")
        cc_sent1 = BeginEndChartConstraints(f"{cc}/sent1/bconst_theta_0.9", f"{cc}/sent1/econst_theta_0.9")

        print("Reading chart constraints for sent2 ...")
        cc_sent2 = BeginEndChartConstraints(f"{cc}/sent2/bconst_theta_0.9", f"{cc}/sent2/econst_theta_0.9")

    num_edges_considered = 0
    num_edges_allowed = 0
    # num_pads = 0

    num_batches = int(MAX_SENTENCES/batchsize)

    for batch in tqdm(range(num_batches), desc="Parsing all sentences"):
        offset = batch * batchsize
        s1 = training_sent1[offset : offset+batchsize]
        s2 = training_sent2[offset : offset+batchsize]

        # lens = [len(s) for s in s1]
        # min_len = min(lens)
        # num_pads += sum([l-min_len for l in lens])
        #
        # lens = [len(s) for s in s2]
        # min_len = min(lens)
        # num_pads += sum([l - min_len for l in lens])

        ops1, oopl1, edgelex1 = convert_sentences(s1, cc_sent1, offset, original_index)
        ops2, oopl2, edgelex2 = convert_sentences(s2, cc_sent2, offset, original_index)

        pr1 = ParsingResult(sentences=s1, ops=ops1, oopl=oopl1, edgelex=edgelex1)
        pr2 = ParsingResult(sentences=s2, ops=ops2, oopl=oopl2, edgelex=edgelex2)

        batched_parses.append((pr1,pr2))

    if MAX_SENTENCES % batchsize > 0:
        offset = num_batches * batchsize
        s1 = training_sent1[offset: MAX_SENTENCES]
        s2 = training_sent2[offset: MAX_SENTENCES]
        ops1, oopl1, edgelex1 = convert_sentences(s1, cc_sent1, offset, original_index)
        ops2, oopl2, edgelex2 = convert_sentences(s2, cc_sent2, offset, original_index)

        pr1 = ParsingResult(sentences=s1, ops=ops1, oopl=oopl1, edgelex=edgelex1)
        pr2 = ParsingResult(sentences=s2, ops=ops2, oopl=oopl2, edgelex=edgelex2)

        batched_parses.append((pr1, pr2))

    print(f"Allowed edges: {num_edges_allowed}/{num_edges_considered} ({100.0*num_edges_allowed/num_edges_considered}%)")
    # print(f"Padding: {num_pads}")

    return batched_parses, training_labels, glove



