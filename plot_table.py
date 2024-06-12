import os
import json
import numpy as np

# 加载记录文件
record_path = './experiments/sz_ridge.json'
if os.path.exists(record_path):
    with open(record_path, 'r') as file:
        records = json.load(file)
else:
    raise FileNotFoundError(f"The record file {record_path} does not exist.")

# Step 1: 整理数据
model_results = {}
for record in records:
    params = record['param']
    model_name = params['model']
    lr = params['lr']
    weight_decay = params['weight_decay']
    param_key = f"lr_{lr}_wd_{weight_decay}"

    if model_name not in model_results:
        model_results[model_name] = {}

    if param_key not in model_results[model_name]:
        model_results[model_name][param_key] = []

    model_results[model_name][param_key].append(record['result'])

# Step 2: 计算每个参数组合的平均值和标准差，并选择最佳参数组合
summary_stats = {}

for model_name, param_results in model_results.items():
    summary_stats[model_name] = {}
    best_auc_mean = -1
    best_param = None
    best_results = None

    for param_key, results in param_results.items():
        auc_values = [result['auc'] for result in results]
        acc_values = [result['accuracy'] for result in results]
        recall_pos_values = [result['recall_pos'] for result in results]

        auc_mean = np.mean(auc_values)
        auc_std = np.std(auc_values)
        acc_mean = np.mean(acc_values)
        acc_std = np.std(acc_values)
        recall_pos_mean = np.mean(recall_pos_values)
        recall_pos_std = np.std(recall_pos_values)

        if auc_mean > best_auc_mean:
            best_auc_mean = auc_mean
            best_param = {
                "lr": lr,
                "weight_decay": weight_decay
            }
            best_results = {
                "acc": [round(acc_mean, 4), round(acc_std, 4)],
                "auc": [round(auc_mean, 4), round(auc_std, 4)],
                "recall_pos": [round(recall_pos_mean, 4), round(recall_pos_std, 4)]
            }

    summary_stats[model_name] = {
        "result": best_results,
        "parameter": best_param
    }

# Step 3: 保存结果到文件
summary_path = './sz_ridge_record.json'
with open(summary_path, 'w') as file:
    json.dump(summary_stats, file, indent=4)

print(f"Summary statistics have been saved to {summary_path}")
latex_table_path = './sz_table.txt'

def generate_latex_table(summary_stats):
    lines = []
    header = "Model Name & Accuracy & AUC & Recall Positive \\\\ \n\\midrule"
    lines.append(header)
    
    for model_name, stats in summary_stats.items():
        acc_mean, acc_std = stats["result"]["acc"]
        auc_mean, auc_std = stats["result"]["auc"]
        recall_pos_mean, recall_pos_std = stats["result"]["recall_pos"]
        
        line = f"{model_name} & ${acc_mean} \\pm {acc_std}$ & ${auc_mean} \\pm {auc_std}$ & ${recall_pos_mean} \\pm {recall_pos_std}$ \\\\"
        lines.append(line)
    
    return "\n".join(lines)

latex_table_content = generate_latex_table(summary_stats)

with open(latex_table_path, 'w') as file:
    file.write(latex_table_content)

print(f"LaTeX table has been saved to {latex_table_path}")
