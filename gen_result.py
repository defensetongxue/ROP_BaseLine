import json
import numpy as np
import os

# Load the data
with open('./experiments/sz_ridge.json') as f:
    records = json.load(f)

best_results = {}

# Define metrics of interest
metrics_of_interest = ["AUC", "Accuracy", "recall_pos"]

# Process each record
for key, value in records.items():
    model_name = value["param"]["model"]
    result = value["result"]
    
    # Calculate mean and standard deviation across all clrs for each metric
    aggregated_results = {}
    for metric in metrics_of_interest:
        metric_values = [float(result[clr][metric]) for clr in result if metric in result[clr]]
        if metric_values:  # Ensure there are valid metric entries
            mean_metric = np.mean(metric_values)
            std_metric = np.std(metric_values)
            # Format mean and std to 4 decimal places as strings
            aggregated_results[metric] = [f"{mean_metric:.4f}", f"{std_metric:.4f}"]

    # Compare and store the best result
    if model_name not in best_results:
        best_results[model_name] = aggregated_results
    else:
        for metric in metrics_of_interest:
            if metric in aggregated_results:
                current_best_mean, current_best_std = [float(x) for x in best_results[model_name].get(metric, ["0", "inf"])]
                new_mean, new_std = [float(x) for x in aggregated_results[metric]]
                if (new_mean > current_best_mean) or (new_mean == current_best_mean and new_std < current_best_std):
                    best_results[model_name][metric] = [f"{new_mean:.4f}", f"{new_std:.4f}"]

# Save the results to JSON
output_json_path = './experiments/result.json'
with open(output_json_path, 'w') as f:
    json.dump(best_results, f, indent=4)

# # Optionally, write to a text file for easy viewing
# output_dir = './experiments'
# output_txt_path = os.path.join(output_dir, 'ridge_table.txt')
# with open(output_txt_path, 'w') as f:
#     for model, results in best_results.items():
#         f.write(f"{model}:\n")
#         for metric, values in results.items():
#             mean, std = values
#             f.write(f"  {metric}: Mean = {mean}, Std = {std}\n")
#         f.write("\n")

print("JSON and text files have been generated with the best results formatted to four decimal places.")
