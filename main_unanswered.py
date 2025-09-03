import csv
import json
import os
from tqdm import tqdm
from time import sleep

from llm import generate


def create_absolute_grading_prompt(instruction, response, reference_answer, rubric):
    """
    Formats the prompt for absolute grading according to the Prometheus model's requirements.

    Args:
        instruction (str): The instruction given to the model that generated the response.
        response (str): The response to be evaluated.
        reference_answer (str): A ground-truth or ideal answer.
        rubric (dict): A dictionary containing the evaluation criteria and score descriptions.

    Returns:
        str: A formatted prompt string.
    """
    # The prompt template is based on the official documentation for the Prometheus model.
    # It provides a structured format for the model to understand the evaluation task.
    prompt = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a one sentence feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (short feedback for criteria) [RESULT] (an integer number between 1 and 5)\nFor example, \"Feedback: The answer is relevant to the question. [RESULT] 5\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer:
{reference_answer}

###Score Rubrics:
[{rubric["criteria"]}]
Score 1: {rubric["score1_description"]}
Score 5: {rubric["score5_description"]}

###Feedback:"""
    return prompt


def load_data(file_path):
    """Loads data from a CSV file into a list of dictionaries."""
    data = []
    with open(file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data.append(dict(row))
    return data


def load_existing_results(model_name):
    file_path = (
        f"spanish_rosie_evals/{model_name.replace('/', '_')}_evaluation_results.json"
    )

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    raise FileNotFoundError(f"No existing results file found for model {model_name}")


def main():
    """Ï
    Main function to run the VLLM inference with the Prometheus model for unanswered questions.
    """

    existing_data_dir = "spanish_rosie_evals"

    file_names = os.listdir(existing_data_dir)
    model_names = [
        x.split("_evaluation_results.json")[0].replace("_", "/")
        for x in file_names
        if x.endswith("_evaluation_results.json") and not x.startswith("prometheus")
    ]

    evaluation_rubrics = [
        {
            "criteria": "Relevance: Is this answer topically relevant?",
            "score1_description": "No",
            "score2_description": "",
            "score3_description": "",
            "score4_description": "",
            "score5_description": "Yes",
        },
        {
            "criteria": "Attributes: All attributions correct?",
            "score1_description": "No",
            "score2_description": "",
            "score3_description": "",
            "score4_description": "",
            "score5_description": "Yes",
        },
        {
            "criteria": "Facts: All facts in answer accounted for in passages?",
            "score1_description": "No",
            "score2_description": "",
            "score3_description": "",
            "score4_description": "",
            "score5_description": "Yes",
        },
        {
            "criteria": "Preference: Do you prefer the reference or model_answer?",
            "score1_description": "Prefer reference",
            "score2_description": "",
            "score3_description": "",
            "score4_description": "",
            "score5_description": "Prefer model_answer",
        },
    ]

    file_path = "spanish_reader_eval_v4_0_with_v2_0_karla_spanish_reader_eval_v4.csv"
    evaluation_data = load_data(file_path)

    # annotated data
    has_all_columns = [
        item
        for item in evaluation_data
        if all(
            item.get(col)
            for col in [
                "Do you prefer passage_1 or model_answer?",
                "All facts in answer accounted for in passages?",
                "All attributions correct?",
                "Is this answer topically relevant?",
            ]
        )
    ]

    for model_name in model_names:
        existing_model_results = load_existing_results(model_name)

        answered_instructions = set(existing_model_results.keys())

        score_mapping = {}

        for item in tqdm(has_all_columns):
            instruction = item.get("question", "")
            response_to_evaluate = item.get("model_answer", "")
            reference_answer = item.get("passage_1", "")

            if instruction in answered_instructions:
                continue

            score_mapping[instruction] = {}
            relevancy = (
                "yes" in item.get("Is this answer topically relevant?", "").lower()
            )
            attribution = "yes" in item.get("All attributions correct?", "").lower()
            facts = (
                "yes"
                in item.get(
                    "All facts in answer accounted for in passages?", ""
                ).lower()
            )
            prefer_model = (
                "model"
                in item.get("Do you prefer passage_1 or model_answer?", "").lower()
            )

            annotation = {
                "relevance": relevancy,
                "attributes": attribution,
                "facts": facts,
                "preference": prefer_model,
            }

            for rubric in evaluation_rubrics:
                absolute_prompt = create_absolute_grading_prompt(
                    instruction, response_to_evaluate, reference_answer, rubric
                )

                generated_text = generate(
                    model_name=model_name, prompt_text=absolute_prompt
                )

                sleep(0.3)

                # parse score to int
                try:
                    feedback, score = generated_text.rsplit("[RESULT]", 1)
                    score = int(score.strip())

                    criteria_key = rubric["criteria"].split(":")[0].lower()

                    score_mapping[instruction][criteria_key] = {
                        "feedback": feedback.strip(),
                        "score": score,
                        "acceptable": score > 3,
                        "human_annotation": annotation[criteria_key],
                    }
                except ValueError:
                    score_mapping[instruction][
                        rubric["criteria"].split(":")[0].lower()
                    ] = {
                        "feedback": generated_text.strip(),
                        "score": None,
                        "acceptable": False,
                    }

        with open(
            f"spanish_rosie_evals/{model_name.replace('/', '_')}_evaluation_results.json",
            "a",
            encoding="utf-8",
        ) as f:
            # Append to existing file or create new if it doesn't exist
            # This will append the new scores to the existing file.
            # For this to work well, we should load existing data first, update it, and then dump it.
            # A simpler approach for this script is to write a *new* file for the new results.
            # Let's stick to the original logic of creating a new file with a new name to avoid corruption.
            # Or, let's load, update, and write back.

            existing_data = {}
            results_file = f"spanish_rosie_evals/{model_name.replace('/', '_')}_evaluation_results.json"
            if os.path.exists(results_file):
                with open(results_file, "r", encoding="utf-8") as f_read:
                    try:
                        existing_data = json.load(f_read)
                    except json.JSONDecodeError:
                        pass  # Keep existing_data empty

            existing_data.update(score_mapping)

            with open(results_file, "w", encoding="utf-8") as f_write:
                json.dump(existing_data, f_write, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
