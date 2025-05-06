import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_histogram_for_label(input_file, target_label):
    generations = []
    total_count = 0
    success_count = 0

    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row['label']
            if label != target_label:
                continue
            total_count += 1
            success = row['success'].lower() == 'true'
            if success:
                success_count += 1
                generations.append(int(row['num_generations']))

    if not generations:
        print(f"No successful trials found for label '{target_label}'.")
        return

    # Plot histogram
    plt.figure(figsize=(10, 6))
    min_gen = min(generations)
    max_gen = max(generations)
    bins = list(range(min_gen - (min_gen % 10), max_gen + 10, 10))
    plt.hist(generations, bins=bins, edgecolor='black')
    plt.xlabel("Number of Generations")
    plt.ylabel("Frequency of Successes")
    plt.title(f"Label: {target_label} | Successes: {success_count} / {total_count}")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    filename = f"{target_label}_success_histogram.png"
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")
    plt.show()

if __name__ == "__main__":
    input_csv = '../data/ablation/generations.csv'
    target_label = sys.argv[1]
    plot_histogram_for_label(input_csv, target_label)
