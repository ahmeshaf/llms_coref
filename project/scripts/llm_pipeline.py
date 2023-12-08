import numpy as np
import openai
import os
import pickle
import re
import typer

from collections import defaultdict
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from .coref_prompt_collections import (
    baseline_output_parser,
    baseline_prompt,
    cot_output_parser,
    eightshot_prompt,
    explanation_prompt,
    fourshot_prompt,
    twoshot_prompt,
    zeroshot_prompt,
)

from .bert.helper import get_context
from .helper import evaluate, ensure_path

# global variables
split_index_map = {"train": 0, "dev": 1, "test": 2}

app = typer.Typer()


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
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")


def extract_answers(strings):
    # Regular expression to extract the "Answer" part
    pattern = r'"Answer":\s*"(\w+)"'
    answers = []
    matches = re.findall(pattern, strings)
    answers.extend(matches)
    return answers


def llm_coref(
    event_pairs: List[Tuple[str, str]],
    mention_map: Dict,
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    cache_file: Path = "/tmp/gpt_pred_coref.pkl",
    gpt_version: str = "gpt-4",
    save_folder: Path = "../../llm_results",
    text_key: str = "marked_doc",
    run_name: str = "baseline",
    temperature: float = 0.7
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
    ensure_path(cache_file)

    if cache_file.exists():
        raw_cache = pickle.load(open(cache_file, "rb"))
    else:
        raw_cache = {}

    result_list = []
    result_dict = defaultdict(dict)

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
                    "Cost": cb.total_cost
                }
                raw_cache[evt_pair] = {"predict": predict, "predict_cost": predict_cost}
                pickle.dump(raw_cache, open(cache_file, "wb"))

        try:
            predict_dict = parser.parse(predict)
            result_list.append(predict)

        except Exception as e:
            print(e)
            # llm might occasionally generate multiple predictions
            # in this case, we take the first one, following the setting in the paper cot.
            answers = extract_answers(str(predict))
            print("total answers detected: ", len(answers))
            predict_dict = {"Answer": answers[0]}

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
    pickle.dump(result_dict, open(save_path, "wb"))
    print(f"Saved prediction results to {save_path}")

    result_list = [1 if r == "True" else 0 for r in result_list]
    return result_list, result_dict


def llm_explanation(
    event_pairs: List[Tuple[str, str]],
    mention_map: Dict,
    prompt: PromptTemplate,
    gpt_version: str = "gpt-4",
    save_folder: Path = "../../llm_explanation",
    text_key: str = "marked_doc",
):
    result_dict = defaultdict(dict)
    llm = ChatOpenAI(temperature=0.0, model=gpt_version, request_timeout=180)
    chain = LLMChain(llm=llm, prompt=prompt)

    for evt_pair in tqdm(event_pairs):
        event1_id = evt_pair[0]
        event2_id = evt_pair[1]

        event1_data = mention_map.get(event1_id)
        event2_data = mention_map.get(event2_id)

        if event1_data is None or event2_data is None:
            continue

        event1_label = event1_data["gold_cluster"]
        event2_label = event2_data["gold_cluster"]
        true_label = "True" if event1_label == event2_label else "False"

        event1 = get_context(event1_data, text_key)
        event2 = get_context(event2_data, text_key)

        answer = chain.run(event1=event1, event2=event2, true_label=true_label)
        format_prompt = chain.prompt.format_prompt(
            event1=event1, event2=event2, true_label=true_label
        )

        result_dict[evt_pair] = {
            "prompt": format_prompt,
            "event1_id": event1_id,
            "event2_id": event2_id,
            "event1": event1,
            "event2": event2,
            "event1_label": event1_label,
            "event2_label": event2_label,
            "eveent1_trigger": event1_data["mention_text"],
            "event2_trigger": event2_data["mention_text"],
            "answer": answer,
        }

    # Save the result_dict
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        save_folder, f"{gpt_version}_explanation_result_{timestamp}.pkl"
    )
    pickle.dump(result_dict, open(save_path, "wb"))
    print(f"Saved explanation results to {save_path}")

    return result_dict


@app.command()
def run_llm_pipeline(
    dataset_folder: str,
    split: str,
    mention_pairs_path: Path,
    results_file: Path,
    debug: bool = False,
    gpt_version: str = "gpt-4",
    gpt_raw_cache_file: Path = "/tmp/gpt.cache",
    experiment_name: str = "baseline",
):
    ensure_path(gpt_raw_cache_file)
    # Set up openai api key
    _ = load_dotenv(find_dotenv())  # Read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Read the mention_map from the dataset_folder
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))

    split_mention_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == split
    ]

    # Get the mention pairs
    mention_pairs = sorted(pickle.load(open(mention_pairs_path, "rb")))
    mention_pairs = [tuple(sorted(p)) for p in mention_pairs]

    # debug
    if debug:
        print("Total mention ids: ", len(split_mention_ids))
        print("Total mention pairs: ", len(mention_pairs))
        mention_pairs = mention_pairs[:5]

    prompt, parser = prompt_and_parser_factory(experiment_name)

    result_list, _ = llm_coref(
        mention_pairs,
        mention_map,
        prompt,
        parser,
        gpt_version,
        save_folder=save_folder,
        run_name=experiment_name,
    )

    # Evaluate the result
    result_array = np.array(result_list)

    scores = evaluate(
        mention_map, split_mention_ids, mention_pairs, similarity_matrix=result_array
    )

    pickle.dump(
        (mention_pairs, result_array, result_array),
        open(results_file, "wb"),
    )

    print(scores)


@app.command()
def run_llm_explanation(
    dataset_folder: str,
    mention_pairs_path: Path,
    debug: bool = False,
    gpt_version: str = "gpt-4",
    save_folder: Path = "../../llm_explanation",
):
    # Set up openai api key
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Read the mention_map from the dataset_folder
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))

    # Get the mention pairs
    mention_pairs = sorted(pickle.load(open(mention_pairs_path, "rb")))

    if debug:
        print("Total mention pairs: ", len(mention_pairs))
        mention_pairs = mention_pairs[:5]

    result_dict = llm_explanation(
        mention_pairs, mention_map, explanation_prompt, gpt_version, save_folder
    )


if __name__ == "__main__":
    app()
