from llm import generate
from pprint import pprint


def judge(input, output, reference, judge_model="openai/gpt-4o-mini"):
    """
    Judge the output of a model based on the input X, candidate output y_hat, and reference r.
    """
    prompt_text = f"""Question: {input}\nProvided Answer: {output}\nReference Answer: {reference}\nEvaluation:\nProvide your response in the following format:\nDecision: [True/False]\nExplanation: [Your brief explanation]"""

    return generate(
        prompt_text=prompt_text,
        system_text="""You are a helpful assistant acting as an impartial judge for a task. You will be given an input and a
        proposed output. Your task is to judge whether the Proposed Answer is correct by comparing it to
        the Reference Answer. If the Proposed Answer is correct, choose ’True’, otherwise choose ’False’.
        Provide a brief explanation for your decision.""",
        model_name=judge_model,
    )


def main():
    # for every entry
    #
    input_text = "What is the capital of France?"
    output_text = "The capital of France is Paris."
    reference_text = "Paris is the capital of France."
    result_1 = judge(input_text, output_text, reference_text)
    result_2 = judge(input_text, "Berlin", reference_text)

    # check if result1 and result2 are the same using match
    lowered_1 = result_1.lower().split()
    lowered_2 = result_2.lower().split()

    decision_1 = True if "true" in lowered_1[0] else False
    decision_2 = True if "true" in lowered_2[0] else False

    explanation_1 = " ".join(lowered_1[1:])
    explanation_2 = " ".join(lowered_2[1:])

    item = {
        "input": input_text,
        "output": output_text,
        "reference": reference_text,
        "judge1": {
            "decision": decision_1,
            "explanation": explanation_1,
        },
        "judge2": {
            "decision": decision_2,
            "explanation": explanation_2,
        },
    }

    if decision_1 != decision_2:
        majority_decision = judge(
            input_text, output_text, reference_text, judge_model="openai/gpt-4o-mini"
        )
        lowered_majority = majority_decision.lower().split()
        majority_decision = True if "true" in lowered_majority[0] else False
        majority_explanation = " ".join(lowered_majority[1:])

        item["majority_decision"] = {
            "decision": majority_decision,
            "explanation": majority_explanation,
        }

    # TODO: record model

    # TODO: write out


if __name__ == "__main__":
    main()
