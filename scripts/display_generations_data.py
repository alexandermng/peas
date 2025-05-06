import csv
from collections import defaultdict

input_file = '../data/ablation/generations.csv'
output_file = 'summary_stats.txt'

stats = defaultdict(lambda: {
    'count': 0,
    'success_true': 0,
    'success_false': 0,
    'total_generations': 0,
    'total_generations_success_true': 0
})

with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label = row['label']
        success = row['success'].lower() == 'true'
        generations = int(row['num_generations'])

        stats[label]['count'] += 1
        stats[label]['total_generations'] += generations
        if success:
            stats[label]['success_true'] += 1
            stats[label]['total_generations_success_true'] += generations
        else:
            stats[label]['success_false'] += 1

with open(output_file, 'w') as f:
    for label in sorted(stats.keys()):
        data = stats[label]
        count = data['count']
        true_count = data['success_true']
        false_count = data['success_false']

        avg_gen_all = data['total_generations'] / count if count > 0 else 0
        avg_gen_true = (
            data['total_generations_success_true'] / true_count
            if true_count > 0 else 0
        )

        f.write(f"Label: {label}\n")
        f.write(f"  Number of Trials: {count}\n")
        f.write(f"  Successes (true): {true_count}\n")
        f.write(f"  Failures (false): {false_count}\n")
        f.write(f"  Average num_generations (all): {avg_gen_all:.2f}\n")
        f.write(f"  Average num_generations (success == true): {avg_gen_true:.2f}\n\n")
