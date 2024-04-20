import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import Counter
import json,os
def calculate_recall(labels, preds, class_id=None):
    """
    Calculate recall for a specified class or for the positive label in a multi-class task.
    
    Args:
    labels (np.array): Array of true labels.
    preds (np.array): Array of predicted labels.
    class_id (int or None): Class ID for which to calculate recall. If None, calculate recall for the positive label.
    
    Returns:
    float: Recall for the specified class or for the positive label.
    """
    if class_id is not None:
        true_class = labels == class_id
        predicted_class = preds == class_id
    else:
        true_class = labels > 0
        predicted_class = preds > 0

    true_positives = np.sum(true_class & predicted_class)
    false_negatives = np.sum(true_class & ~predicted_class)

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall

class Metrics:
    def __init__(self, header="Main",num_class=2 ):
        self.reset()
        self.num_class=num_class
        self.header=header
    def reset(self):
        self.accuracy = 0
        self.auc = 0
        self.recall_1 = 0
        self.recall_2 = 0
        self.recall_3 = 0
        self.recall_pos=0

    def update(self, predictions, probs, targets):
        self.accuracy = accuracy_score(targets, predictions)
        if self.num_class==2:
            self.auc= roc_auc_score(targets,predictions)
        else:
            self.auc = roc_auc_score(targets, probs, multi_class='ovr')
        self.recall_0 = calculate_recall(targets, predictions, class_id=0)
        self.recall_1 = calculate_recall(targets, predictions, class_id=1)
        self.recall_2 = calculate_recall(targets, predictions, class_id=2)
        self.recall_3 = calculate_recall(targets, predictions, class_id=3)
        self.recall_pos=calculate_recall(targets,predictions)

    def __str__(self):
        return (f"[{self.header}] "
                f"Acc: {self.accuracy:.4f}, Auc: {self.auc:.4f}, "
                f"Recall1: {self.recall_1:.4f}, Recall2: {self.recall_2:.4f}, "
                f"Recall3: {self.recall_3:.4f}, RecallPos: {self.recall_pos:.4f} ")
    
    def _store(self,key, split_name,param, save_path='./record.json'):
        res = {
        "Accuracy": f"{self.accuracy:.4f}",
        "AUC": f"{self.auc:.4f}",
        "recall_pos":  f"{self.recall_pos:.4f}",  # Assuming this is a single value, not formatted
        "0_recall": f"{self.recall_0:.4f}",
        "1_recall": f"{self.recall_1:.4f}",
        "2_recall": f"{self.recall_2:.4f}",
        "3_recall": f"{self.recall_3:.4f}",
        }

        # Check if the file exists and load its content if it does
        if os.path.exists(save_path):
            with open(save_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = {}

        # Append the new data
        if key not in existing_data:
            existing_data[key]={
                "param":param,
                "result":{}
            }
        existing_data[key]["result"][split_name]=res

        # Save the updated data back to the file
        with open(save_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
            
            