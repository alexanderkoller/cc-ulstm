import time

import torch
import torchtext
from nltk.tree import Tree
from torch.optim import Adam

from model import MaillardSnliModel


def train(batched_parses, training_labels, dev_batched_parses, dev_labels, glove, device, batchsize, hd, lr, num_epochs, show_zero_ops, experiment):
    BATCHSIZE = batchsize
    NUM_EPOCHS = num_epochs

    # set up model and optimizer
    model = MaillardSnliModel(hd, 100, 3, glove, device).to(device)
    model.set_inv_temperature(10.0)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        total_loss = 0

        for batch in range(len(batched_parses)):
            print(f"\nbatch {batch}")
            start = time.time()

            lab = torch.LongTensor(training_labels[batch * BATCHSIZE: (batch + 1) * BATCHSIZE]).to(device)
            pr1, pr2 = batched_parses[batch]

            if show_zero_ops:
                all_ops = 0
                zero_ops = 0
                for x in pr1.ops:
                    for y in x:
                        for z in y:
                            all_ops += 1
                            if z == (0, 0):
                                zero_ops += 1

                print(f"zero ops: {zero_ops}/{all_ops} ({100*zero_ops/all_ops}%)")

            mid = time.time()
            # print("forward")
            predictions = model(pr1.sentences, pr1.ops, pr1.oopl, pr2.sentences, pr2.ops, pr2.oopl)

            loss = criterion(predictions, lab)
            total_loss += loss.item()

            after_forward = time.time()
            # print("backward")

            loss.backward()  # -> check that gradients that should be non-zero are
            optimizer.step()

            end = time.time()

            print(f"loss in batch {batch}: {loss.item()}")
            print(f"convert: {mid-start}, forward: {after_forward-mid}, backward: {end-after_forward}")

            del loss  # to free it up before next iteration


            # update temperature
            new_inv_temp = (float(epoch) + batch/len(batched_parses)) * 100.0 + 1.0 # from Maillard source code, not paper
            model.set_inv_temperature(new_inv_temp)

        # sys.exit(0)

        mean_loss = total_loss/len(batched_parses)
        print(f"\n\n=== mean loss after epoch {epoch}: {mean_loss} ===\n")

        dev_accuracy = eval_dev(model, dev_batched_parses, dev_labels, glove)
        sys.exit(0)

        if experiment is not None:
            experiment.log_metric("loss", mean_loss, step=epoch + 1)

    return mean_loss



def eval_dev(model, dev_batched_parses, dev_labels, glove):
    correct_labels = 0
    total_labels = 0

    for batch in range(len(dev_batched_parses)):
        pr1, pr2 = dev_batched_parses[batch]
        bs = len(pr1.sentences)
        predictions, trees1, trees2 = model.predict(pr1.sentences, pr1.ops, pr1.oopl, pr2.sentences, pr2.ops, pr2.oopl)

        for i in range(bs):
            print(decode_tree(trees1[i], glove))


def map(tree, fn):
    mapped_children = [map(tree[i], fn) for i in range(len(tree))]
    mapped_label = fn(tree.label())
    new_label = tree.label() if mapped_label is None else mapped_label
    return Tree(new_label, mapped_children)


def decode_tree(tree, glove):
    def lookup(word_id):
        if type(word_id) is int:
            if word_id < len(glove.itos):
                return glove.itos[word_id]
            else:
                return "*UNK*"
        else:
            return None

    return map(tree, lookup)