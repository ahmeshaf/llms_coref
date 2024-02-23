import json
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import openai
import typer
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser
from tqdm import tqdm

from .helper import ensure_path
from .special_prompt_collections import metaphor_prompt_multi, metaphor_prompt_single

app = typer.Typer()
random.seed(42)


class JsonParser:
    def __init__(self):
        pass

    def parse(self, output):
        return json.loads(output)


def prompt_and_parser_factory(
    prompt_type: str,
) -> Tuple[PromptTemplate, StructuredOutputParser]:
    if prompt_type == "meta_single":
        return metaphor_prompt_single, JsonParser()
    elif prompt_type == "meta_multi":
        return metaphor_prompt_multi, JsonParser()
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")


def llm_meta(
    flatten_doc_sent_map: Dict,
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    cache_dir: Path = "/tmp/gpt_meta/",
    gpt_version: str = "gpt-4",
    save_folder: Path = "../outputs/",
    run_name: str = "meta_single",
    temperature: float = 0.7,
    split: str = "test",
) -> Dict:
    # Prepare the cache file
    cache_file = os.path.join(cache_dir, f"{gpt_version}_{run_name}_cache.pkl")
    ensure_path(cache_file)
    cache_file = Path(cache_file)

    if cache_file.exists():
        raw_cache = pickle.load(open(cache_file, "rb"))
    else:
        raw_cache = {}

    # Initialize the result dict
    result_dict = defaultdict(dict)

    # Initialize the LLM
    llm = ChatOpenAI(temperature=temperature, model=gpt_version, request_timeout=180)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate
    for key, value in tqdm(flatten_doc_sent_map.items()):
        if value is None:
            continue

        format_prompt = chain.prompt.format_prompt(
            sentence=value["sentence"], trigger_list=",".join(value["mention_texts"])
        )
        if key in raw_cache:
            predict = raw_cache[key]["predict"]
        else:
            with get_openai_callback() as cb:
                predict = chain.run(
                    sentence=value["sentence"],
                    trigger_list=", ".join(value["mention_texts"]),
                )

                predict_cost = {
                    "Total": cb.total_tokens,
                    "Prompt": cb.prompt_tokens,
                    "Completion": cb.completion_tokens,
                    "Cost": cb.total_cost,
                }
                raw_cache[key] = {
                    "predict": predict,
                    "predict_cost": predict_cost,
                    "format_prompt": format_prompt,
                }
                pickle.dump(raw_cache, open(cache_file, "wb"))
        try:
            predict_dict = parser.parse(predict)
            predict_dict["format_prompt"] = format_prompt

        except Exception as e:
            print(e)

        result_dict[key] = predict_dict

    # Save the result_dict
    save_folder = os.path.join(save_folder, run_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        save_folder,
        f"{gpt_version}_{run_name}_{split}_{timestamp}.pkl",
    )
    pickle.dump(result_dict, open(save_path, "wb"))
    print(f"Saved prediction results to {save_path}")

    return result_dict


@app.command()
def run_llm_meta_pipeline(
    dataset_folder: str,
    split: str,
    debug: bool = False,
    gpt_version: str = "gpt-4",
    gpt_raw_cache_dir: Path = "/tmp/gpt_meta/",
    save_folder: Path = "../outputs/",
    experiment_name: str = "meta_single",
):
    ensure_path(gpt_raw_cache_dir)
    # # Set up OpenAI API key
    # _ = load_dotenv(find_dotenv())  # Read local .env file
    # openai.api_key = os.environ["OPENAI_API_KEY"]

    # Read the mention_map from the dataset_folder
    doc_sent_map = pickle.load(open(dataset_folder + "/doc_sent_map.pkl", "rb"))

    # Flatten the doc_sent_map
    flattened_dict = defaultdict(dict)
    for doc_key, sentences in doc_sent_map.items():
        for sent_key, sent_data in sentences.items():
            new_key = (doc_key, sent_data["sent_id"])
            flattened_dict[new_key] = sent_data["sentence"]

    prompt, parser = prompt_and_parser_factory(experiment_name)

    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    flattened_dict = defaultdict(dict)

    # Get the sentence and mention_texts of the specific split
    for key, value in mention_map.items():
        if value["split"] == split and value["men_type"] == "evt":
            flattened_dict[(value["doc_id"], value["sentence_id"])]["sentence"] = value[
                "sentence"
            ]
            if (
                flattened_dict[(value["doc_id"], value["sentence_id"])].get(
                    "mention_texts"
                )
                is None
            ):
                flattened_dict[(value["doc_id"], value["sentence_id"])][
                    "mention_texts"
                ] = [value["mention_text"]]
            else:
                flattened_dict[(value["doc_id"], value["sentence_id"])][
                    "mention_texts"
                ].append(value["mention_text"])

    # Debug
    if debug:
        print("Total sentences: ", len(flattened_dict))
        first_5_items = list(flattened_dict.items())[:5]
        flattened_dict = dict(first_5_items)

    result_dict = llm_meta(
        flattened_dict,
        prompt,
        parser,
        gpt_version=gpt_version,
        save_folder=save_folder,
        run_name=experiment_name,
        cache_dir=gpt_raw_cache_dir,
        split=split,
    )
    # Debug
    if debug:
        print(result_dict)


if __name__ == "__main__":
    app()
