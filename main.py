import socket

from comet_ml import Experiment

import argparse
import json
import os
import sys
import time
from collections import namedtuple

import nltk
# import spacy
import torch
import torchtext
from torch.optim import Adam
from tqdm import tqdm

import data
import train
from chart_constraints import AllAllowedChartConstraints, BeginEndChartConstraints
from model import SequentialChart, MaillardSnliModel
from util import get_num_lines

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('--epochs', default='10', type=int)
parser.add_argument('--bs', default='20', type=int)
parser.add_argument('--hidden-dim', default='10', type=int)
parser.add_argument('--lr', default='0.01', type=float)
parser.add_argument('--show-zero-ops', action='store_true')
parser.add_argument('--limit', default='100', type=int)
parser.add_argument('--maxlen', default='20', type=int) # skip sentences longer than this
parser.add_argument('--comet', default=None, type=str)
parser.add_argument('--cc', default=None, type=str)
parser.add_argument('--sort', action='store_true') # sort by length
# parser.add_argument('--init-temperature', default='1.0', type=float)

args = parser.parse_args()


COMET_API_KEY = args.comet

if COMET_API_KEY is not None:
    # Record experiment in Comet

    experiment  = Experiment(api_key=COMET_API_KEY, project_name="CC-ULSTM")

    hyper_params = {
        "batch_size": args.bs,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "device": str(device),
        "hidden_dim": args.hidden_dim,
        "limit": args.limit,
        "cc": args.cc,
        "sort": args.sort
    }
    hyper_params["hostname"] = socket.gethostname() or "(undefined)"

    experiment.log_multiple_params(hyper_params)
else:
    experiment = None

# def get_data(train_file, batchsize, limit, sort, cc):
batched_parses, training_labels, glove = data.get_data("data/snli_1.0/snli_1.0_train.jsonl", args.bs, args.limit, args.maxlen, args.sort, args.cc)
dev_batched_parses, dev_labels, _ = data.get_data("data/snli_1.0/snli_1.0_dev.jsonl", args.bs, args.limit, args.maxlen, args.sort, None, glove=glove, mode="dev") # TODO - use CC for dev sentences too

#def train(batched_parses, training_labels, glove, device, batchsize, hd, lr, num_epochs, initial_temperature, show_zero_ops, experiment):
final_mean_loss = train.train(batched_parses, training_labels, dev_batched_parses, dev_labels, glove, device, args.bs, args.hidden_dim, args.lr, args.epochs, args.show_zero_ops, experiment)

