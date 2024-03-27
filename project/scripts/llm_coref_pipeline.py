import json
import os
import pickle
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openai
import typer
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser
from tqdm import tqdm

from .bert.helper import get_context
from .coref_prompt_collections import (
    baseline_output_parser,
    baseline_prompt,
    cot_output_parser,
    eightshot_prompt,
    fourshot_prompt,
    twoshot_prompt,
    zeroshot_prompt,
)
from .helper import ensure_dir, ensure_path, evaluate, filter_topic_aligned_cross_doc_pairs

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
    if prompt_type == "baseline":
        return baseline_prompt, baseline_output_parser
    elif prompt_type == "zeroshot":
        return zeroshot_prompt, cot_output_parser
    elif prompt_type == "twoshot":
        return twoshot_prompt, cot_output_parser
    elif prompt_type == "fourshot":
        return fourshot_prompt, cot_output_parser
    elif prompt_type == "eightshot":
        return eightshot_prompt, cot_output_parser
    elif prompt_type == "davidson":
        return davidson_prompt, JsonParser()
    elif prompt_type == "quine":
        return quine_prompt, JsonParser()
    elif prompt_type == "amr":
        return amr_prompt, JsonParser()
    elif prompt_type == "arg":
        return arg_prompt, JsonParser()
    elif prompt_type == "adv":
        return adv_prompt, JsonParser()
    elif prompt_type == "adv_v2":
        return adv_prompt_v2, JsonParser()
    elif prompt_type == "sum":
        return summarization_prompt, JsonParser()
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")


def extract_answers(strings):
    # Regular expression to extract the "Answer" part
    pattern = r'"Answer":\s*"(\w+)"'
    answers = []
    matches = re.findall(pattern, strings)
    answers.extend(matches)
    return answers


def extract_answers_beta(strings):
    # Regular expression to extract the "Answer" part
    pattern1 = r'"*Answer"*\s*:\s*(true|false)'
    pattern2 = r"is\s*(true|false)"

    # First, try to find matches with pattern1
    matches = re.findall(pattern1, strings, re.IGNORECASE)
    if not matches:
        # If no matches found with pattern1, try with pattern2
        matches = re.findall(pattern2, strings, re.IGNORECASE)
    # Extracting the true|false part
    answers = [match for match in matches if match]
    return answers


def process_response(tuple_id, response):
    try:
        lines = response["predict"].strip().split("\n")
        if "}" not in lines[-1]:
            lines.append("}")
        structured_lines = []
        for line in lines:
            line = line.strip('", ')
            if line == "":
                continue
            if len(line.split(":")) == 1:
                structured_lines.append(line)
            elif len(line.split(":")) >= 2:
                splits = line.split(":")
                key = splits[0]
                value = ":".join(splits[1:])
                key = key.strip(' ",')
                key = f"{json.dumps(key)}"
                value = value.strip(' ",')
                value = f"{json.dumps(value)},"
                structured_lines.append(key + ":" + value)
            else:
                print("Unknown line: ", line)

        structured_lines[-2] = structured_lines[-2].strip(",")
        structured_json = "\n".join(structured_lines)
        structured_json = json.loads(structured_json)
        return structured_json
    except Exception as e:
        print(f"An error occurred at {id}:", e)
        return None


def llm_coref(
        event_pairs: List[Tuple[str, str]],
        mention_map: Dict,
        prompt: PromptTemplate,
        parser: StructuredOutputParser,
        cache_dir: Path = "/tmp/gpt_pred_coref/",
        gpt_version: str = "gpt-4",
        save_folder: Path = "../../llm_results",
        text_key: str = "marked_doc",
        run_name: str = "baseline",
        temperature: float = 0.7,
) -> Dict:
    """
    Predict coreference using the LLM (Language Model) for provided event pairs.

    Parameters
    ----------
    event_pairs : list of tuple of str
        A list of event pairs where each pair is represented as a tuple of event IDs.
    mention_map : dict
        A dictionary mapping event IDs to their respective data.
        Data should include 'bert_sentence' representing the event's sentence.
    prompt : PromptTemplate
        A template for the LLM prompt to query the model.
    parser : StructuredOutputParser
        A parser to extract structured output from the LLM's response.
    gpt_version : str, optional
        The version of the GPT model to use. Default is "gpt-4".
    cache_file: Path
    text_key: str
    run_name: str
    save_folder : str, optional
        Folder to save the prediction results. Default is "../../llm_results".
    temperature: float


    Returns
    -------
    list of int
        A list of prediction results, where 1 represents 'True' and 0 represents 'False'.
    dict
        A dictionary containing detailed prediction results for each event pair,
        including event IDs, event sentences, and parsed predictions.
    """
    cache_file = os.path.join(cache_dir, f"{gpt_version}_{run_name}_cache.pkl")
    ensure_path(cache_file)
    cache_file = Path(cache_file)

    if cache_file.exists():
        raw_cache = pickle.load(open(cache_file, "rb"))
    else:
        raw_cache = {}

    result_list = []
    result_dict = defaultdict(dict)
    # result_dict = not_found.add((m1, m2))

    # initialize the llm
    llm = ChatOpenAI(
        temperature=temperature, model=gpt_version, request_timeout=180
    )  # Set the request_timeout to 180 seconds
    chain = LLMChain(llm=llm, prompt=prompt)

    # predict
    for evt_pair in tqdm(event_pairs):
        event1_id = evt_pair[0]
        event2_id = evt_pair[1]

        event1_data = mention_map.get(event1_id)
        event2_data = mention_map.get(event2_id)

        true_label = str(event1_data["gold_cluster"] == event2_data["gold_cluster"])

        if event1_data is None or event2_data is None:
            continue

        event1_text = get_context(event1_data, text_key)
        event2_text = get_context(event2_data, text_key)

        format_prompt = chain.prompt.format_prompt(event1=event1_text, event2=event2_text)

        if evt_pair in raw_cache:
            predict = raw_cache[evt_pair]["predict"]
        else:
            with get_openai_callback() as cb:
                predict = chain.run(event1=event1_text, event2=event2_text)
                predict_cost = {
                    "Total": cb.total_tokens,
                    "Prompt": cb.prompt_tokens,
                    "Completion": cb.completion_tokens,
                    "Cost": cb.total_cost,
                }
                raw_cache[evt_pair] = {"predict": predict, "predict_cost": predict_cost}
                pickle.dump(raw_cache, open(cache_file, "wb"))
        try:
            predict_dict = parser.parse(predict)
            print("++============ PROMPT =============+++")
            print(format_prompt.text)
            print("\n\n================ RESPONSE ===================", "\n\n", str(predict))
            print("\n\n================ True Label ===================", "\n\n", str(true_label), "\n\n")

        except Exception as e:
            print(e)
            # llm might occasionally generate multiple predictions
            # in this case, we take the first one, following the setting in the paper cot.
            answers = extract_answers(str(predict))
            print("total answers detected: ", len(answers))
            if answers:
                predict_dict = {"Answer": answers[0]}
            else:
                print("No answer detected, set to False")
                predict_dict = {"Answer": "False"}

        result_list.append(predict_dict["Answer"])
        result_dict[evt_pair] = {
            "event1_id": event1_id,
            "event2_id": event2_id,
            "event1": event1_text,
            "event2": event2_text,
            "event1_trigger": event1_data["mention_text"],
            "event2_trigger": event2_data["mention_text"],
            "predict_raw": predict,
            "predict_dict": predict_dict,
        }

    # Save the result_dict
    save_folder = os.path.join(save_folder, run_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        save_folder,
        f"{gpt_version}_{run_name}_predict_result_{timestamp}.pkl",
    )
    # pickle.dump(result_dict, open(save_path, "wb"))
    # print(f"Saved prediction results to {save_path}")

    result_array = []
    str_count = 0
    bool_count = 0
    other_count = 0
    for r in result_list:
        if isinstance(r, str):
            if r.lower() == "true":
                r = 1
            elif r.lower() == "false":
                r = 0
            else:
                print("Unknow String: ", r)
                r = 0
            str_count += 1
        elif isinstance(r, bool):
            if r:
                r = 1
            else:
                r = 0
            bool_count += 1
        else:
            print("Unknown type: ", type(r))
            print("r: ", r)
            r = 0
            other_count += 1
        result_array.append(r)
    print("Total: ", len(result_list))
    print("str_count: ", str_count)
    print("bool_count: ", bool_count)
    print("other_count: ", other_count)
    result_array = np.array(result_array)
    return result_array, result_dict


@app.command()
def run_llm_pipeline(
        mention_map_path: str,
        split: str,
        mention_pairs_path: Path,
        results_file: Path,
        debug: bool = False,
        gpt_version: str = "gpt-4",
        gpt_raw_cache_dir: Path = "/tmp/gpt_pred_coref/",
        save_folder: Path = "../../llm_results",
        experiment_name: str = "baseline",
        text_key: str = "marked_sentence",
):
    ensure_dir(gpt_raw_cache_dir / f"a.l")
    # Set up openai api key
    _ = load_dotenv(find_dotenv())  # Read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Read the mention_map from the dataset_folder
    mention_map = pickle.load(open(mention_map_path, "rb"))

    split_mention_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == split
    ]

    # Get the mention pairs
    mention_pairs = sorted(pickle.load(open(mention_pairs_path, "rb")))
    mention_pairs = [tuple(sorted(p)) for p in mention_pairs]

    # Filter out same sentence
    print("before filtering: ", len(mention_pairs))
    mention_pairs = filter_topic_aligned_cross_doc_pairs(mention_pairs, mention_map)
    print("after filtering: ", len(mention_pairs))

    # debug
    if debug:
        print("Total mention ids: ", len(split_mention_ids))
        print("Total mention pairs: ", len(mention_pairs))
        random.shuffle(mention_pairs)
        # mention_pairs = mention_pairs[:5]
        pos_pairs = [(m1, m2) for m1, m2 in mention_pairs if
                     mention_map[m1]["gold_cluster"] == mention_map[m2]["gold_cluster"]]
        neg_pairs = [(m1, m2) for m1, m2 in mention_pairs if
                     mention_map[m1]["gold_cluster"] != mention_map[m2]["gold_cluster"]]
        mention_pairs = pos_pairs[:5] + neg_pairs[:5]

    prompt, parser = prompt_and_parser_factory(experiment_name)

    result_list, result_dict = llm_coref(
        mention_pairs,
        mention_map,
        prompt,
        parser,
        gpt_version=gpt_version,
        save_folder=save_folder,
        run_name=experiment_name,
        text_key=text_key,
        cache_dir=gpt_raw_cache_dir,
    )

    # Evaluate the result
    result_array = np.array(result_list)

    scores = evaluate(
        mention_map, split_mention_ids, mention_pairs, similarity_matrix=result_array
    )

    # Save the results
    pickle.dump(
        (mention_pairs, result_list, result_array),
        open(results_file, "wb"),
    )

    # Debug
    # if debug:
    #     print(result_dict)

    print(scores)


if __name__ == "__main__":
    app()
