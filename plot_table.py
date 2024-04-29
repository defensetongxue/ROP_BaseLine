import json
import os

# Load JSON data
with open('./experiments/fs_ridge.json') as f:
    record = json.load(f)

best_result = {}
Using_Matric = ["AUC", "Accuracy", "recall_pos"]

# Process each entry
for key, entry in record.items():
    model_name = entry["param"]["model"]
    results = entry["result"]

    # Aggregate results by averaging metric values across different clr_* entries
    aggregated_results = {metric: 0 for metric in Using_Matric}
    count = 0
    for result_key, metrics in results.items():
        count += 1
        for metric in Using_Matric:
            aggregated_results[metric] += float(metrics[metric])

    for metric in aggregated_results:
        aggregated_results[metric] /= count  # Calculate average

    # Update best result for the model based on AUC
    if model_name not in best_result or best_result[model_name]["AUC"] < aggregated_results["AUC"]:
        best_result[model_name] = aggregated_results

# Generate a LaTeX table
output_dir = './experiments'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'ridge_table.txt')

# LaTeX table formatting
header = " & ".join(["Model"] + Using_Matric) + " \\\\ \\hline"
rows = []
for model, metrics in best_result.items():
    row = [model] + [f"{metrics[metric]:.4f}" for metric in Using_Matric]
    rows.append(" & ".join(row) + " \\\\")

table_content = "\n".join(rows)
latex_table = f"\\begin{{table}}[h]\n\\centering\n\\begin{{tabular}}{{{'|'.join(['c'] * (len(Using_Matric) + 1))}}}\n\\hline\n{header}\n{table_content}\n\\end{{tabular}}\n\\caption{{Best model results}}\n\\end{{table}}"

# Save to file
with open(output_file, 'w') as f:
    f.write(latex_table)

print(f"LaTeX table has been generated and saved to {output_file}.")
