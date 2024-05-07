import json

# Load JSON data
with open('./experiments/result.json', 'r') as f:
    record = json.load(f)

# Path to save the LaTeX table
save_path = './experiments/sz_ridge_table.txt'

# Start writing the LaTeX table
table_content = "\\begin{table}[ht]\n"
table_content += "\\centering\n"
table_content += "\\caption{Summary of Model Performance. Each value represents the mean performance with its corresponding standard deviation, formatted as mean $\\pm$ std.}\n"
table_content += "\\label{tab:model_performance}\n"
table_content += "\\begin{tabular}{lccc}\n"
table_content += "\\toprule\n"
table_content += "Model Name & Accuracy & AUC & Recall Positive \\\\\n"
table_content += "\\midrule\n"

# Iterate over each model and add data to the table
for model_name, metrics in record.items():
    # Escape underscores in model names to avoid LaTeX formatting issues
    safe_model_name = model_name.replace('_', '\\_')
    # accuracy = f"${metrics['Accuracy'][0]} \\pm {metrics['Accuracy'][1]}$"
    accuracy = f"${metrics['Accuracy'][0]}$"
    # auc = f"${metrics['AUC'][0]} \\pm {metrics['AUC'][1]}$"
    auc = f"${metrics['AUC'][0]}$"
    # recall_positive = f"${metrics['recall_pos'][0]} \\pm {metrics['recall_pos'][1]}$"
    recall_positive = f"${metrics['recall_pos'][0]}$"
    table_content += f"{safe_model_name} & {accuracy} & {auc} & {recall_positive} \\\\\n"

# Close the table structure
table_content += "\\bottomrule\n"
table_content += "\\end{tabular}\n"
table_content += "\\end{table}\n"

# Save the LaTeX table content to a file
with open(save_path, 'w') as f:
    f.write(table_content)

print("LaTeX table saved to", save_path)
