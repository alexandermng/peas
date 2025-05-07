#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt


def plot_trial(csv_file, trial_id_to_plot):
    generations = []
    avg_fitness = []
    max_fitness = []

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_id = int(row['trial_id'])
            if trial_id == trial_id_to_plot:
                generations.append(int(row['generation']))
                avg_fitness.append(float(row['avg_fitness']))
                max_fitness.append(float(row['max_fitness']))

    if not generations:
        print(f"No data found for trial_id {trial_id_to_plot}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, color='blue', label='Avg Fitness', marker='o')
    plt.plot(generations, max_fitness, color='red', label='Max Fitness', marker='o')

    plt.title(f"Fitness Progression for Trial ID {trial_id_to_plot}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    y_max = max(max_fitness)
    plt.ylim(0, y_max * 1.05)  # Add 5% padding above the max
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"trial_{trial_id_to_plot}_plot.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot fitness progression for a specific trial.")
    parser.add_argument(
        "-c", "--csv",
        type=str,
        default="../data/ablation/data.csv",
        help="Path to the CSV file containing trial data (default: ../data/ablation/data.csv)"
    )
    parser.add_argument(
        "-t", "--trial",
        type=int,
        default=3,
        help="Trial ID to plot (default: 3)"
    )

    args = parser.parse_args()
    plot_trial(args.csv, args.trial)
