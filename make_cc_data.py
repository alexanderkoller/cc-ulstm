import json
import sys

import nltk

from tqdm import tqdm

from util import get_num_lines

train_file = "data/snli_1.0/snli_1.0_train.jsonl"
with open(train_file) as f:
    with open("tokens_sent2.txt", "w") as ftokens:
        with open("tags_sent2.txt", "w") as ftags:
            for line in tqdm(f, desc="Reading training sentences", total=get_num_lines(train_file)):
                j = json.loads(line)

                t1 = nltk.Tree.fromstring(j["sentence2_parse"])
                tokens, tags = zip(* t1.pos())
                tokens = [t.lower() for t in tokens]
                print("\t".join(tokens), file=ftokens)
                print("\t".join(tags), file=ftags)

