import time
import sys
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
        print(f"\n\n=== mean training loss after epoch {epoch}: {mean_loss} ===")

        dev_accuracy = eval_dev(epoch, model, dev_batched_parses, dev_labels, glove)
        print(f"=== dev accuracy: {dev_accuracy} ===\n")

        if experiment is not None:
            experiment.log_metric("training loss", mean_loss, step=epoch + 1)
            experiment.log_metric("dev accuracy", dev_accuracy, step=epoch+1)

    return mean_loss



def eval_dev(epoch, model, dev_batched_parses, dev_labels, glove):
    correct_labels = 0
    total_labels = 0

    def lookup(word_id):
        if type(word_id) is int:
            if word_id < 0:
                return "**UNK**"
            elif word_id < len(glove.itos):
                return glove.itos[word_id]
            else:
                return "*UNK*"
        else:
            return None

    with open(f"trees_after_{epoch}.txt", "w") as f:
        for batch in range(len(dev_batched_parses)):
            pr1, pr2 = dev_batched_parses[batch]
            bs = len(pr1.sentences)
            print(f"max sentence len: {max(len(s) for s in pr1.sentences)}")
            predictions, trees1, trees2 = model.predict(pr1.sentences, pr1.ops, pr1.oopl, pr1.edgelex, pr2.sentences, pr2.ops, pr2.oopl, pr2.edgelex)
            gold_labels = dev_labels[batch * bs: (batch + 1) * bs]

            # write parse trees to file
            for i in range(bs):
                print("", file=f)
                print(" ".join([lookup(id) for id in pr1.sentences[i]]), file=f)
                map(trees1[i], lookup).pprint(margin=10000000, stream=f)

            # calculate dev prediction accuracy
            assert len(gold_labels) == len(predictions), f"dev_labels: {len(dev_labels)} != predictions: {len(predictions)}"

            correct = [i for i in range(len(gold_labels)) if gold_labels[i] == predictions[i]]
            correct_labels += len(correct)
            total_labels += len(gold_labels)

            del predictions

    return correct_labels/total_labels




def map(tree, fn):
    mapped_children = [map(tree[i], fn) for i in range(len(tree))]
    mapped_label = fn(tree.label())
    new_label = tree.label() if mapped_label is None else mapped_label
    return Tree(new_label, mapped_children)

