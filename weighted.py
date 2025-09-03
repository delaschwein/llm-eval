from llm import generate
import json
from typing import Any

def judge(input, output, reference, judge_model="openai/gpt-4o-mini"):
    prompt_text = f"Please anylyze the following responses:\nContext: {input}\nAI Response: {output}\nGold Response: {reference}\nEvaluation:\nProvide your evaluation in the following JSON format:\n{{'acceptable': <True/False>, 'explanation': <Your brief explanation>}}"

    return generate(
        prompt_text=prompt_text,
        system_text="""You are an AI judge evaluating the quality of an AI-generated response compared to a gold standard response. Your task is to determine if the AI response is an acceptable output based on the following criteria:\n1. Compare the factual content of both responses.\n2. Check if the AI response includes facts that are also present in the gold response.\n3. The AI response can have additional facts not present in the gold response.\n4. The AI response should not miss any critical or supporting facts from the gold response.\n5. The AI response can miss trivial facts from the gold response.""",
        model_name=judge_model,
    )


def main():
    # for every entry
    #
    input_text = "What is the capital of France?"
    output_text = "The capital of France is Paris."
    reference_text = "Paris is the capital of France."
    result = judge(input_text, output_text, reference_text)

    print(type(result))
    print(result)

    item: dict[str, Any] = {
        "input": input_text,
        "output": output_text,
        "reference": reference_text
    }

    try:
        result_json = json.loads(result)

        item["judge"] = {
            "acceptable": result_json.get("acceptable", False),
            "explanation": result_json.get("explanation", "")
        }
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

        item["judge"] = {
            "raw_result": result,
        }
    # TODO: record model

    # TODO: write out


if __name__ == "__main__":
    main()
