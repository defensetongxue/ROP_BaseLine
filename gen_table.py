import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_averages(results):
    metrics = ['accuracy', 'auc', 'recall_1', 'recall_2', 'recall_3', 'recall_pos', 'average_recall']
    averages = {metric: 0 for metric in metrics}
    count = 0

    for result in results.values():
        for metric in metrics:
            averages[metric] += result.get(metric, 0)
        count += 1
    
    if count > 0:
        averages = {metric: value / count for metric, value in averages.items()}
    
    return averages

def generate_markdown_table(data):
    markdown_table = "| Model | Accuracy | AUC | Recall 1 | Recall 2 | Recall 3 | Recall Pos | Average Recall |\n"
    markdown_table += "|-------|----------|-----|----------|----------|----------|------------|----------------|\n"
    
    for key, value in data.items():
        model_name = key.split('_')[0]
        avg_results = calculate_averages(value["result"])
        row = f"| {model_name} | {avg_results['accuracy']:.4f} | {avg_results['auc']:.4f} | {avg_results['recall_1']:.4f} | {avg_results['recall_2']:.4f} | {avg_results['recall_3']:.4f} | {avg_results['recall_pos']:.4f} | {avg_results['average_recall']:.4f} |\n"
        markdown_table += row

    return markdown_table

# Path to your JSON file
file_path = './record.json'
data = load_data(file_path)
markdown_table = generate_markdown_table(data)
print(markdown_table)
