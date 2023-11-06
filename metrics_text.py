#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import re
from collections import defaultdict
from typing import List
import numpy as np
import nltk
from nltk import edit_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import json

# Download NLTK WordNet resource (if not already downloaded)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    print("Downloading NLTK WordNet resource...")
    nltk.download('wordnet')

def compute_metrics(pred, gt, minlen=4):
    metrics = {}
    if len(pred) < minlen or len(gt) < minlen:
        return metrics
    metrics["edit_dist"] = edit_distance(pred, gt) / max(len(pred), len(gt))
    reference = gt.split()
    hypothesis = pred.split()
    metrics["bleu"] = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method1)
    try:
        from nltk.translate.meteor_score import meteor_score
        metrics["meteor"] = meteor_score([reference], hypothesis)
    except ImportError:
        metrics["meteor"] = np.nan
    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["precision"] = len(reference.intersection(hypothesis)) / len(hypothesis)
    metrics["recall"] = len(reference.intersection(hypothesis)) / len(reference)
    metrics["f_measure"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    return metrics

def get_input():
    predicted_text = input("Enter predicted text: ")
    ground_truth_text = input("Enter ground truth text: ")
    return predicted_text, ground_truth_text

if __name__ == "__main__":
    predicted_text, ground_truth_text = get_input()
    
    # Calculate metrics
    metrics = compute_metrics(predicted_text, ground_truth_text)
    
    # Display results
    print("Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Organize data into a JSON structure
    data = {
        "predictions": [predicted_text],
        "ground_truths": [ground_truth_text]
    }
    
    # Save the data as a JSON file
    with open("data.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    print("Data saved as data.json")


# In[ ]:




