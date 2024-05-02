import json
import os
import numpy as np

# Load data
with open('./experiments/fs_ridge.json') as f:
    records = json.load(f)

best_result = {}

# Process each record to find the best result based on AUC
for key, value in records.items():
    model_name = value["param"]["model"]
    results = value["result"]
    
    # Calculate mean metrics across splits for each record
    metrics = {"AUC": [], "Accuracy": [], "recall_pos": []}
    for split in results.values():
        for metric in metrics.keys():
            if metric in split:
                metrics[metric].append(float(split[metric]))
    
    # Calculate mean of each metric
    mean_metrics = {metric: np.mean(vals) for metric, vals in metrics.items()}
    
    if model_name not in best_result or best_result[model_name]["AUC"] < mean_metrics["AUC"]:
        best_result[model_name] = mean_metrics

# Calculate mean and standard deviation for each metric for each model
result_stats = {model: {metric: [np.mean(values), np.std(values)]
                        for metric, values in res.items()}
                for model, res in best_result.items()}

# Save to JSON
output_json_path = './experiments/result.json'
with open(output_json_path, 'w') as f:
    json.dump(result_stats, f, indent=4)

# Save to text file
output_dir = './experiments'
output_txt_path = os.path.join(output_dir, 'ridge_table.txt')
with open(output_txt_path, 'w') as f:
    for model, metrics in result_stats.items():
        f.write(f"{model}:\n")
        for metric, values in metrics.items():
            mean, std = values
            f.write(f"  {metric}: Mean = {mean:.4f}, Std = {std:.4f}\n")
    f.write("\n")

print("Analysis completed and results saved.")
