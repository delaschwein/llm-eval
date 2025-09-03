import json


def analyze_results(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    criteria = {}

    for question, results in data.items():
        for criterion, values in results.items():
            if criterion not in criteria:
                criteria[criterion] = {
                    "total": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "acceptable": 0,
                }

            criteria[criterion]["total"] += 1

            model_prediction = values.get("acceptable", False)
            human_label = values.get("human_annotation", False)

            if model_prediction:
                criteria[criterion]["acceptable"] += 1

            if model_prediction and human_label:
                criteria[criterion]["tp"] += 1
            elif model_prediction and not human_label:
                criteria[criterion]["fp"] += 1
            elif not model_prediction and human_label:
                criteria[criterion]["fn"] += 1

    print("Evaluation Statistics:")
    for criterion, counts in criteria.items():
        total = counts["total"]
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        accuracy = (counts["acceptable"] / total) * 100 if total > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"\nCriterion: {criterion}")
        print(f"  - Accuracy: {accuracy:.2f}%")
        print(f"  - Precision: {precision:.2f}")
        print(f"  - Recall: {recall:.2f}")
        print(f"  - F1-score: {f1:.2f}")


import os
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stats.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        sys.exit(1)

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            print(f"Analyzing {file_path}...")
            analyze_results(file_path)
