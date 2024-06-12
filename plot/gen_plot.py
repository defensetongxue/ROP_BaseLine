import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_metrics(data):
    selected = ["resnet18", "resnet50", "vitPool", "efficientnetB7"]
    colors = [(51/255, 57/255, 91/255), (93/255, 116/255, 162/255), (196/255, 216/255, 242/255), (242/255, 232/255, 227/255)]
    metrics = ['accuracy', 'auc', 'recall_pos', 'average_recall']
    
    rows = []
    for key, value in data.items():
        model_name = key.split('_')[0]
        if model_name not in selected:
            continue
        for metric in metrics:
            avg_result = round(np.mean([value["result"][str(i)].get(metric, 0) for i in range(1, 5)]) * 100, 1)
            rows.append({'model': model_name, 'metric': metric, 'score': avg_result, 'color': colors[selected.index(model_name)]})

    return pd.DataFrame(rows)

def plot_metrics(df, save_path):
    models = df['model'].unique()
    metrics = df['metric'].unique()
    colors = dict(zip(models, [(51/255, 57/255, 91/255), (93/255, 116/255, 162/255), (196/255, 216/255, 242/255), (242/255, 232/255, 227/255)]))

    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.2  # Width of the bars
    x = np.arange(len(metrics))  # Label locations

    for i, model in enumerate(models):
        scores = df[df['model'] == model]['score']
        ax.bar(x + i * width, scores, width, label=model, color=colors[model])

    ax.set_title('Model Metrics Comparison')
    ax.set_xlabel('Metric Name')
    ax.set_ylabel('Metric Score (%)')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
# Path to your JSON file and save path
file_path = './record.json'
save_path = './save_plot.png'

data = load_data(file_path)
df = extract_metrics(data)
plot_metrics(df, save_path)
