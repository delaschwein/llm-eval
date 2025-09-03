import json
import argparse


def compute_ensemble(file1, file2, file3):
    """
    Computes the ensemble judgment from three model result files.

    Args:
        file1 (str): Path to the first model result file (JSON).
        file2 (str): Path to the second model result file (JSON).
        file3 (str): Path to the third model result file (JSON).

    Returns:
        dict: A dictionary containing the ensemble judgments.
    """
    with open(file1, "r") as f:
        data1 = json.load(f)
    with open(file2, "r") as f:
        data2 = json.load(f)
    with open(file3, "r") as f:
        data3 = json.load(f)

    ensemble_results = {}
    all_questions = sorted(list(data1.keys()))

    for question in all_questions:
        ensemble_results[question] = {}
        for aspect in ["relevance", "attributes", "facts", "preference"]:
            scores = []
            acceptables = []
            human_annotations = []

            for data in [data1, data2, data3]:
                if question in data and aspect in data[question]:
                    score = data[question][aspect].get("score")
                    if isinstance(score, (int, float)):
                        scores.append(score)

                    acceptable = data[question][aspect].get("acceptable")
                    if isinstance(acceptable, bool):
                        acceptables.append(acceptable)

                    human_annotation = data[question][aspect].get("human_annotation")
                    if isinstance(human_annotation, bool):
                        human_annotations.append(human_annotation)

            if scores:
                avg_score = sum(scores) / len(scores)
            else:
                avg_score = None

            if acceptables:
                # Majority vote for acceptable
                ensemble_acceptable = acceptables.count(True) >= 2
            else:
                ensemble_acceptable = False

            ensemble_human_annotation = any(human_annotations)

            ensemble_results[question][aspect] = {
                "average_score": avg_score,
                "acceptable": ensemble_acceptable,
                "human_annotation": ensemble_human_annotation,
                "individual_scores": scores,
                "individual_acceptables": acceptables,
                "individual_human_annotations": human_annotations,
            }

    return ensemble_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ensemble judgments from three model result files."
    )
    parser.add_argument("file1", help="Path to the first model result file.")
    parser.add_argument("file2", help="Path to the second model result file.")
    parser.add_argument("file3", help="Path to the third model result file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output JSON file. If not provided, prints to stdout.",
    )

    args = parser.parse_args()

    ensemble_data = compute_ensemble(args.file1, args.file2, args.file3)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(ensemble_data, f, indent=4)
        print(f"Ensemble results saved to {args.output}")
    else:
        print(json.dumps(ensemble_data, indent=4))
