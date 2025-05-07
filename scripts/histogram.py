#!/usr/bin/env python3
import csv
import argparse
from collections import defaultdict
from os import path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot histogram of generations for a given label.")
parser.add_argument("label", help="Label to filter data by")
parser.add_argument("-i", "--infile", default="../data/ablation/generations.csv", help="Path to CSV input file")
parser.add_argument("-o", "--outdir", default=".", help="Path to output directory")
parser.add_argument("-s", "--successes_only", action="store_true", help="Only include successful trials in the histogram")
args = parser.parse_args()

generations = []
success_count = 0
total_count = 0

with open(args.infile, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label = row['label']
        if label != args.label:
            continue

        total_count += 1
        success = row['success'].lower() == 'true'
        gen = int(row['num_generations'])

        if args.successes_only:
            if success:
                generations.append(gen)
                success_count += 1
        else:
            generations.append(gen)
            if success:
                success_count += 1

min_gen = min(generations)
max_gen = max(generations)
bins = list(range(min_gen - (min_gen % 10), max_gen + 10, 10))

plt.figure(figsize=(10, 6))

if args.successes_only:
    plt.hist(generations, bins=bins, edgecolor='black')
else:
    counts, bin_edges, patches = plt.hist(generations, bins=bins, edgecolor='black')
    if patches and bin_edges[-1] >= 100:
        patches[-1].set_facecolor('red')  # assume some failure if over 100 gens

plt.xlabel("Number of Generations")
plt.ylabel("Frequency")
experiment = ""
if args.outdir:
    experiment = path.basename(args.outdir)
title = f"{experiment} {args.label} | Successes: {success_count} / {total_count} trials"
plt.title(title)
plt.grid(True)
plt.tight_layout()

filename = path.join(args.outdir, f"hist_{args.label}.png")
plt.savefig(filename)
print(f"Histogram saved as {filename}")
plt.show()
