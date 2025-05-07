#!/usr/bin/env python3
import csv
import argparse
from os import path, makedirs
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_histogram(title: str, generations: list[int], successes_only: bool = False):
    """
    Plots a histogram of the number of generations.

    Args:
        title (str): Title of the plot.
        generations (list[int]): List of generation counts.
        success_count (int): Number of successful trials.
        label (str): The label for the histogram.
        successes_only (bool): If True, only include successful trials in the histogram.

    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    min_gen = min(generations)
    max_gen = max(generations)
    bins = list(range(min_gen - (min_gen % 10), max_gen + 10, 10))

    plt.figure(figsize=(10, 6))

    if successes_only:
        plt.hist(generations, bins=bins, edgecolor='black')
    else:
        counts, bin_edges, patches = plt.hist(generations, bins=bins, edgecolor='black')
        if patches and bin_edges[-1] >= 100:
            # Color the last bin red, as long as it's above 100 gens (assuming no experiment will be over 100 gens with zero failures)
            patches[-1].set_facecolor('red')

    plt.xlabel("Number of Generations")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    return plt


def main(args):
    """
    Main function to parse the CSV file and generate histograms for each label.

    Args:
        args: Parsed command-line arguments.
    """
    experiment_dir = path.join("data", args.experiment)
    infile = path.join(experiment_dir, "generations.csv")
    outdir = experiment_dir

    label_generations = defaultdict(list)  # generations keyed by label
    label_success_counts = defaultdict(int)  # success counts keyed by label

    # Parse the CSV file
    with open(infile, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row['label']
            success = row['success'].lower() == 'true'
            gen = int(row['num_generations'])

            label_generations[label].append(gen)
            if success:
                label_success_counts[label] += 1

    # Histogram for each label
    print('Generating histograms:')
    for label, generations in label_generations.items():
        experiment: str = f"{args.experiment} {label}"
        title = f"{experiment.replace('_', ' ').title()} | Successes: {label_success_counts[label]} / {len(generations)} trials"
        plot = plot_histogram(
            title,
            generations,
            args.successes_only,
        )

        filename = path.join(outdir, f"hist_{label}.png")
        plot.savefig(filename)
        print(f"\t{label:<25} --> {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histograms of generations for each label.")
    parser.add_argument("experiment", help="Name of the experiment (used as the directory under 'data/')")
    parser.add_argument("-s", "--successes_only", action="store_true", help="Only include successful trials in the histogram")
    args = parser.parse_args()

    main(args)
