import csv
import matplotlib.pyplot as plt
from collections import defaultdict

trials = defaultdict(list)

with open("../data/speciation_range/data.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trial_id = int(row["trial_id"])
        row["threshold"] = float(row["threshold"])
        row["generation"] = int(row["generation"])
        row["max_fitness"] = float(row["max_fitness"])
        trials[trial_id].append(row)

threshold_stats = defaultdict(lambda: {
    "total": 0,
    "successful": 0,
    "success_generations": []
})

for trial_id, rows in trials.items():
    threshold = rows[0]["threshold"]
    max_gen = max(row["generation"] for row in rows)
    max_fitness = max(row["max_fitness"] for row in rows)

    threshold_stats[threshold]["total"] += 1
    if max_fitness >= 1.0:
        threshold_stats[threshold]["successful"] += 1
        threshold_stats[threshold]["success_generations"].append(max_gen)

thresholds = sorted(threshold_stats.keys())
success_rates = []
avg_generations = []

for t in thresholds:
    data = threshold_stats[t]
    rate = data["successful"] / data["total"]
    success_rates.append(rate)

    if data["success_generations"]:
        avg_gen = sum(data["success_generations"]) / len(data["success_generations"])
    else:
        avg_gen = 0
    avg_generations.append(avg_gen)

# Plotting with dual y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = "tab:green"
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Average Generation (successful trials)", color=color1)
ax1.plot(thresholds, avg_generations, color=color1, marker='o', label="Avg Generations")
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()  # Create second y-axis
color2 = "tab:blue"
ax2.set_ylabel("Success Rate", color=color2)
ax2.plot(thresholds, success_rates, color=color2, marker='x', linestyle='--', label="Success Rate")
ax2.tick_params(axis='y', labelcolor=color2)

plt.title("Threshold vs Avg Generation and Success Rate")
fig.tight_layout()
plt.grid(True)

filename = f"spec_range.png"
plt.savefig(filename)
print(f"Plot saved as {filename}")

plt.show()
