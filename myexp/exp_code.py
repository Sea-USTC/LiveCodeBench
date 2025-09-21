import os
import json

from lcb_runner.lm_styles import LanguageModel
from lcb_runner.runner.parser import get_args
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.benchmarks import CodeGenerationProblem
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.runner.scenario_router import sort_and_extract_save_results, get_metrics
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.lm_styles import LMStyle


from datasets import load_dataset


def get_codeqwen_question_template_answer(question: CodeGenerationProblem):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n<|im_end|>\n"
    else:
        prompt += f"Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    prompt +=  question.experience
    return prompt

def format_prompt_generation(
    question: CodeGenerationProblem, LanguageModelStyle: LMStyle
) -> str:
    if LanguageModelStyle == LMStyle.CodeQwenInstruct:
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\n"
        prompt += f"{get_codeqwen_question_template_answer(question)}"
        return prompt


    raise NotImplementedError(
        f"LanguageModelStyle {LanguageModelStyle} not implemented"
    )


def load_code_generation_dataset(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", revision='refs/pr/7', name=release_version)
    dataset = [CodeGenerationProblem(**p) for p in dataset[1]]  # type: ignore
    dataset[0].experience=("Dynamic Programming Code Generation Reminder:"
                            "When generating DP code, pay special attention to these common oversights:"
                            "State transition completeness: Ensure you consider ALL possible previous states that can contribute to the current state. Think carefully about the relationship between the current round of state updates and the previous round."
                            "Conditional logic accuracy: When translating the thought process into code, be careful with if-else statement nesting and branching logic to ensure each subcase follows the correct conditional path. ")
    print(f"Loaded {len(dataset)} problems")
    return dataset


def build_prompt_benchmark(
    args,
) -> tuple[
    list[CodeGenerationProblem],
    callable,
]:
    benchmark = load_code_generation_dataset(
            args.release_version,
            start_date=args.start_date,
            end_date=args.end_date
        )
    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    format_prompt = format_prompt_generation
    return benchmark, format_prompt

def extract_code(model_output: str):
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    # return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])

def combine_results(
    results: list[list[str]],
    model: LanguageModel,
):
    combined_results = [
        (
            outputs_list,
            [extract_code(output, model.model_style) for output in outputs_list],
        )
        for outputs_list in results
    ]

    return combined_results


def main():
    args = get_args()
    model = LanguageModelStore[args.model]
    benchmark, format_prompt = build_prompt_benchmark(args)
    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    old_save_results = []
    remaining_benchmark = benchmark
    runner = build_runner(args, model)
    results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)

    combined_results = combine_results(
        results, model
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    if args.evaluate:
        
        metrics = get_metrics(args.scenario, args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])
        old_eval_all_results = []
        old_eval_results = []

        if metrics:
            metadatas = metrics[2]
        else:
            metadatas = [[] for _ in benchmark]
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list, metadata=meta
            )
            for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                benchmark, combined_results, graded, metadatas
            )
        ]
        if metrics and old_eval_results:
            old_eval_results
            metrics[2] = old_eval_results[2] + metrics[2]

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)



if __name__ == "__main__":
    main()