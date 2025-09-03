import json


def main():
    model_choices = [
        ["gpt-3.5-turbo", "anthropic/claude-3.5-sonnet", "openai/gpt-4"],
        ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "openai/gpt-4"],
        [
            "meta-llama/llama-3-8b-instruct",
            "meta-llama/llama-3-70b-instruct",
            "mistralai/mixtral-8x7b-instruct",
        ],
        ["gpt-3.5-turbo", "prometheus", "meta-llama/llama-3-8b-instruct"],
    ]

    for models in model_choices:
        judge_one = models[0]
        judge_two = models[1]
        arbitration = models[2]

        judge_one_file = (
            f"spanish_rosie_evals/{judge_one.replace('/', '_')}_evaluation_results.json"
        )
        judge_two_file = (
            f"spanish_rosie_evals/{judge_two.replace('/', '_')}_evaluation_results.json"
        )
        arbitration_file = f"spanish_rosie_evals/{arbitration.replace('/', '_')}_evaluation_results.json"

        output_file = f"spanish_rosie_evals/dafe_{judge_one.replace('/', '_')}_{judge_two.replace('/', '_')}_{arbitration.replace('/', '_')}.json"

        with open(judge_one_file, "r", encoding="utf-8") as f:
            data1 = json.load(f)

        with open(judge_two_file, "r", encoding="utf-8") as f:
            data2 = json.load(f)

        with open(arbitration_file, "r", encoding="utf-8") as f:
            data3 = json.load(f)

        output = {}

        for question in data1.keys():
            if question not in data2 or question not in data3:
                raise ValueError(f"Question {question} not found in all files.")

            data1_question = data1[question]
            data2_question = data2[question]
            data3_question = data3[question]

            current_item = {}

            for aspect in ["relevance", "attributes", "facts", "preference"]:
                human_annotations = data1_question.get(aspect, {}).get(
                    "human_annotation", False
                )

                if data1_question.get(aspect, {}).get(
                    "acceptable"
                ) == data2_question.get(aspect, {}).get("acceptable"):
                    current_item[aspect] = {
                        "acceptable": data1_question.get(aspect, {}).get(
                            "acceptable", False
                        ),
                        "human_annotation": human_annotations,
                    }
                else:
                    current_item[aspect] = {
                        "acceptable": data3_question.get(aspect, {}).get(
                            "acceptable", False
                        ),
                        "human_annotation": human_annotations,
                    }

            output[question] = current_item

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
