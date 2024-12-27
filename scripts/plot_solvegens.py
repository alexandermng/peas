#!/usr/bin/env python3

import os
import argparse
import json
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def plot_trials(trials: list) -> Figure:

    fig, ax = plt.subplots(figsize=(10, 6))

    solvegens = [get_num_gens(t) for t in trials]

    bins = 20
    ax.hist(solvegens, bins=bins, color="lightblue", edgecolor="black")

    ax.set_title("Histograms of Generations per Trial")
    ax.set_xlabel("Number of Generations")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)

    return fig


def get_num_gens(trial: str) -> int:
    resultsfile = os.path.join(trial, "results.json")
    results = {}
    try:
        with open(resultsfile, "r") as file:
            results = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{resultsfile}' not found.")
        return -1
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON. {e}")
        return -1
    assert results
    assert "num_generations" in results
    return results["num_generations"]


def main(args):
    # Load manifest
    try:
        with open(args.manifest, "r") as file:
            manifest = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON. {e}")
        return

    trials = manifest[args.problem]  # list of trial log directories
    fig = plot_trials(trials)

    # Save and show the plot
    fig.savefig(args.output)
    print(f"Plot saved to '{args.output}'.")
    # if args.quiet == True:
    #     return
    # fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manifest', action='store',
                        default="data/manifest.json", help="Manifest file")
    parser.add_argument('-p', '--problem', action='store',
                        default="sum3", help="Problem to check")  # make required?
    parser.add_argument('-o', '--output', action='store',
                        default="solve_generations.png", help="Output file")
    # parser.add_argument('-q', '--quiet', action='store_true',
    #                     default=False, help="Do not show plot.")
    args = parser.parse_args()
    main(args)
