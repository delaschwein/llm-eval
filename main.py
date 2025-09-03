import csv

from llm import generate
import json
from tqdm import tqdm
from time import sleep

def create_absolute_grading_prompt(instruction, response, reference_answer, rubric):
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
[{rubric['criteria']}]
Score 1: {rubric['score1_description']}
Score 5: {rubric['score5_description']}

###Feedback:"""
    return prompt

def load_data(file_path):
    """Loads data from a CSV file into a list of dictionaries."""
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data.append(dict(row))
    return data

def main():
    """Ï
    Main function to run the VLLM inference with the Prometheus model.
    """

    model_names = ["openai/gpt-3.5-turbo"]

    evaluation_rubrics = [
        {
            "criteria": "Relevance: Is this answer topically relevant?",
            "score1_description": "No",
            "score2_description": "",
            "score3_description": "",
            "score4_description": "",
            "score5_description": "Yes"
        },
        {
            "criteria": "Attributes: All attributions correct?",
            "score1_description": "No",
            "score2_description": "",
            "score3_description": "",
            "score4_description": "",
            "score5_description": "Yes"
        },
        {
            "criteria": "Facts: All facts in answer accounted for in passages?",
            "score1_description": "No",
            "score2_description": "",
            "score3_description": "",
            "score4_description": "",
            "score5_description": "Yes"
        },
        {
            "criteria": "Preference: Do you prefer the reference or model_answer?",
            "score1_description": "Prefer reference",
            "score2_description": "",
            "score3_description": "",
            "score4_description": "",
            "score5_description": "Prefer model_answer"
        }
    ]

    file_path = 'spanish_reader_eval_v4_0_with_v2_0_karla_spanish_reader_eval_v4.csv'
    evaluation_data = load_data(file_path)

    # annotated data
    has_all_columns = [
        item for item in evaluation_data
        if all(item.get(col) for col in [
            'Do you prefer passage_1 or model_answer?',
            'All facts in answer accounted for in passages?',
            'All attributions correct?',
            'Is this answer topically relevant?'
        ])
    ]


    for model_name in model_names:
        score_mapping = {}

        for item in tqdm(has_all_columns):
            instruction = item.get('question', '')
            response_to_evaluate = item.get('model_answer', '')
            reference_answer = item.get('passage_1', '')
            score_mapping[instruction] = {}
            relevancy = 'yes' in item.get('Is this answer topically relevant?', '').lower()
            attribution = 'yes' in item.get('All attributions correct?', '').lower()
            facts = 'yes' in item.get('All facts in answer accounted for in passages?', '').lower()
            prefer_model = 'model' in item.get('Do you prefer passage_1 or model_answer?', '').lower()

            annotation = {
                "relevance": relevancy,
                "attributes": attribution,
                "facts": facts,
                "preference": prefer_model
            }


            for rubric in evaluation_rubrics:
                absolute_prompt = create_absolute_grading_prompt(
                    instruction,
                    response_to_evaluate,
                    reference_answer,
                    rubric
                )

                generated_text = generate(model_name=model_name, prompt_text=absolute_prompt)

                sleep(0.3)

                # parse score to int
                try:
                    feedback, score = generated_text.rsplit("[RESULT]", 1)
                    score = int(score.strip())

                    criteria_key = rubric['criteria'].split(":")[0].lower()

                    score_mapping[instruction][criteria_key] = {
                    "feedback": feedback.strip(),
                    "score": score,
                    "acceptable": score > 3,
                    "human_annotation": annotation[criteria_key]
                    }
                except ValueError:
                    score_mapping[instruction][rubric['criteria'].split(":")[0].lower()] = {
                        "feedback": generated_text.strip(),
                        "score": None,
                        "acceptable": False
                    }


        with open(f"spanish_rosie_evals/{model_name.replace('/', '_')}_evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(score_mapping, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    main()
