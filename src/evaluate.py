import pandas as pd
import argparse
import json
import os
import torch
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.pipelines.base')

from src.utils import get_timestamp, parse_model_list
from src.models import LanguageModel, MODELS

def main():
    current_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ARGUMENT PARSER
    parser = argparse.ArgumentParser(description="LLM Evaluation on OUS, returns raw data")
    parser.add_argument(
        "--dataset", default=os.path.join(current_parent_dir, "data/ous.json"), help="dataset path"
    )
    parser.add_argument(
        "--models",
        default=[
        # "gpt-3.5-turbo-0125",
        # "gpt-3.5-turbo-1106",
        # "gpt-4-turbo",
        # "gemini-pro",
        # 'mistral-7b',
        #'gemma-7b',
        'falcon-7b',
        #'yi-6b',
        ],
        type=parse_model_list,
        help="List of models to evaluate, separated by comma. E.g. 'gpt-3.5-turbo-0125, gpt-3.5-turbo-1106'"
    )
    parser.add_argument(
        "--instructions",
        default=os.path.join(current_parent_dir, "data/instructions.json"),
        type=str,
        help="Path to instructions file"
    )
    parser.add_argument(
        "--system-prompt",
        default=os.path.join(current_parent_dir, "data/system_prompt.json"),
        type=str,
        help="Path to system prompt file"
    )
    parser.add_argument(
        "--eval-temp",
        default=1.0,
        type=float,
        help="Temperature for sampling. Default: 1.0 (when max is 2), 0.5 (when max is 1)"
    )
    parser.add_argument(
        "--eval-max-tokens",
        default=200,
        type=int,
        help="Max. number of tokens per completion"
    )
    parser.add_argument(
        "--num-samples", default=10, type=int, help="Nb. of samples per statement"
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(current_parent_dir, "outputs/raw/"), type=str, help="Output directory"
    )

    args = parser.parse_args()

    # SETUP
    data_file = args.dataset
    instructions_file = args.instructions
    system_prompt_file = args.system_prompt
    models = args.models
    num_samples = args.num_samples

    output_dir = args.output_dir
    max_tokens = args.eval_max_tokens
    temperature = args.eval_temp

    run_evaluation(data_file, instructions_file, system_prompt_file, models, num_samples, output_dir, temperature, max_tokens)

def run_evaluation(data_file, instructions_file, system_prompt_file, models, num_samples, output_dir, temperature, max_tokens=None):

    assert os.path.exists(data_file), f"Data file not found: {data_file}"
    assert os.path.exists(instructions_file), f"Instructions file not found: {instructions_file}"
    assert os.path.exists(system_prompt_file), f"System prompt file not found: {system_prompt_file}"
    assert os.path.exists(output_dir), f"Output directory not found: {output_dir}"
    for model_name in models:
        assert model_name in MODELS.keys(), f"Model not found: {model_name}"

    # load data
    with open(data_file, encoding="utf-8") as f:
        dataset = json.load(f)
        dataset = dataset["questions"]

    with open(instructions_file, encoding="utf-8") as f:
        instructions = json.load(f)

    with open(system_prompt_file, encoding="utf-8") as f:
        system_prompt = json.load(f)
        system_prompt = system_prompt["system"]
    current = 0
    # RUN EVALUATION
    for model_name in models:
        model = LanguageModel.create(model_name, temperature=temperature)
        print("Running model:", model._model_name)
        data = {
            "instruction" : [],
            "statement" : [],
            "answer" : [],
            "temperature" : [],
            "timestamp" : [],
            "nationality" : [],
            "company" : []
        }

        for instruction in tqdm(list(instructions.keys()), desc='Instruction ', position=0):
            for k, statement in enumerate(tqdm(dataset, desc='Statement ', position=1, leave=False)):
                for i in range(num_samples):
                    input_data = {
                        "system": system_prompt,
                        "instruction": instructions[instruction],
                        "dataset": statement
                    }

                    data["instruction"].append(instruction)
                    data["statement"].append(k)
                    data["answer"].append(model.invoke(input_data))
                    data["temperature"].append(model._temperature)
                    data["timestamp"].append(get_timestamp())
                    data["nationality"].append(model.nationality)
                    data["company"].append(model.company)

                    file_name = os.path.join(output_dir, f"{model.model_id}_raw_data.csv")
                    pd.DataFrame(data).to_csv(file_name, sep=";", index=False)
                    current += 1
                    if(current%10==0):
                        print(f"{current} calls done")
        print('Saving raw data to: ', file_name)
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
