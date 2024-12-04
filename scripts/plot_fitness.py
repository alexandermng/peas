#!/usr/bin/env python3

import argparse
import json
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def plot_data(data) -> Figure:
    max_fitnesses = data["max_fitnesses"]
    avg_fitnesses = data["avg_fitnesses"]
    num_generations = data["num_generations"]
    assert len(max_fitnesses) == num_generations
    assert len(avg_fitnesses) == num_generations

    fig, ax = plt.subplots(figsize=(10, 6))
    # TODO ability to pass in fig

    generations = list(range(num_generations))  # x axis
    ax.plot(generations, max_fitnesses, label="Max Fitness", color="blue", marker="o")
    ax.plot(generations, avg_fitnesses, label="Average Fitness", color="green", marker="x")

    ax.set_title("Fitness Over Generations")
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(True)

    return fig


def main(args):

    # Load data from JSON
    try:
        with open(args.input, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON. {e}")
        return

    # Valid data
    assert "max_fitnesses" in data
    assert "avg_fitnesses" in data
    assert "num_generations" in data

    fig = plot_data(data)

    # Save and show the plot
    fig.savefig(args.output)
    print(f"Plot saved to '{args.output}'.")
    # if args.quiet == True:
    #     return
    # fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='store',
                        default="results.json", help="Input file")
    parser.add_argument('-o', '--output', action='store',
                        default="fitness.png", help="Output file")
    # parser.add_argument('-q', '--quiet', action='store_true',
    #                     default=False, help="Do not show plot.")
    args = parser.parse_args()
    main(args)
