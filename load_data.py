from datasets import load_dataset
import os
import json

dataset_labeled = load_dataset("qiaojin/PubmedQA", "pqa_labeled", split="train")
dataset_unlabeled = load_dataset("qiaojin/PubmedQA", "pqa_unlabeled", split="train")
dataset_artificial = load_dataset("qiaojin/PubmedQA", "pqa_artificial", split="train")

dataset_labeled_file = "data/pubmedqa_labeled.json"
dataset_unlabeled_file = "data/pubmedqa_unlabeled.json"
dataset_artificial_file = "data/pubmedqa_artificial.json"

def save_dataset(ds, file_path):
    data_list = [dict(ds[i]) for i in range(len(ds))] 
    with open(file_path, "w") as f:
        json.dump(data_list, f, indent=2)

save_dataset(dataset_labeled, dataset_labeled_file)
save_dataset(dataset_unlabeled, dataset_unlabeled_file)
save_dataset(dataset_artificial, dataset_artificial_file)

print("All datasets have been converted to JSON and stored")
