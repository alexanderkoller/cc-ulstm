import argparse
import re

parser = argparse.ArgumentParser(description='Analyze hyperparameters.')
parser.add_argument('slurm_logfile', type=str)
parser.add_argument('hyperparameters', default='hyperparameters.tsv', type=str)

args = parser.parse_args()

pattern = re.compile(r"\s*(\d+): result: loss=([.0-9]+), time=([.0-9]+)")

results = {}
with open(args.slurm_logfile, "r") as fl:
    for line in fl:
        m = pattern.match(line)
        if m:
            taskid = int(m.group(1))
            results[taskid] = (m.group(2), m.group(3))
            print(f"{taskid}: {results[taskid]}")

print(f"batchsize\tLR\thidden_dim\tloss\ttime")
with open(args.hyperparameters, "r") as fh:
    for i, line in enumerate(fh):
        loss, time = results[i]
        print(f"{line.strip()}\t{loss}\t{time}")

