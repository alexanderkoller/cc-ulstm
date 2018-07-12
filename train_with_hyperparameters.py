import csv
import os
import time

import torch

import data
import train

task_id = int(os.environ['SLURM_PROCID']) if "SLURM_PROCID" in os.environ else 0

with open('hyperparameters.tsv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter='\t')
    hp_choices = [row for row in readCSV]

row = hp_choices[task_id]
bs, lr, hd = int(row[0]), float(row[1]), int(row[2])

print(bs, lr, hd)

LIMIT=100
EPOCHS=1
TEMP=1.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def get_data(train_file, batchsize, limit, sort, cc):
batched_parses, training_labels, glove = data.get_data("data/snli_1.0/snli_1.0_train.jsonl", bs, LIMIT, True, "data/snli_1.0/cc")

start_time = time.time()
#def train(batched_parses, training_labels, glove, device, batchsize, hd, lr, num_epochs, initial_temperature, show_zero_ops, experiment):
final_mean_loss = train.train(batched_parses, training_labels, glove, device, bs, hd, lr, EPOCHS, TEMP, False, None)
end_time = time.time()

print(f"result: loss={final_mean_loss}, time={end_time-start_time}")
