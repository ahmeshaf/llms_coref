# Standard Library Imports
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openai
import typer
from dotenv import find_dotenv, load_dotenv

# Local/Custom Imports
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser
from tqdm import tqdm

from .coref_prompt_collections import (
    baseline_output_parser,
    baseline_prompt,
    cot_output_parser,
    eightshot_prompt,
    fourshot_template,
    zeroshot_prompt,
)
from .nn_method.helper import get_context
from .heuristic import biencoder_nn, get_lh_pairs, lh
from .prediction import evaluate

# global variables
split_index_map = {"train": 0, "dev": 1, "test": 2}

app = typer.Typer()


# helper functions
def read(key, response):
    return get_coref_infos("%s" % key, "%s" % response, False, False, True)


def prompt_and_parser_factory(
    prompt_type: str,
) -> Tuple[PromptTemplate, StructuredOutputParser]:
    if prompt_type == "baseline":
        return baseline_prompt, baseline_output_parser
    elif prompt_type == "zeroshot":
        return zeroshot_prompt, cot_output_parser
    elif prompt_type == "twoshot":
        return fourshot_template, cot_output_parser
    elif prompt_type == "fourshot":
        return fourshot_template, cot_output_parser
    elif prompt_type == "eightshot":
        return eightshot_prompt, cot_output_parser
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")


# def run_llm_lh(
#     dataset: str,
#     split: str,
#     prompt: PromptTemplate,
#     parser: StructuredOutputParser,
#     lh_threshold: float = 0.5,
# ) -> Tuple[List, Dict]:
#     """
#     Run the LLM (Language Model) with a linguistic heuristic (LH) for a specified dataset and split.

#     Parameters
#     ----------
#     dataset : str
#         The name of the dataset.
#     split : str
#         The data split (e.g., 'train', 'test', 'dev').
#     prompt : PromptTemplate
#         A template for the LLM prompt to query the model.
#     parser : StructuredOutputParser
#         A parser to extract structured output from the LLM's response.
#     lh_threshold : float, optional
#         The threshold value for the linguistic heuristic. Defaults to 0.5.

#     Returns
#     -------
#     list
#         A list of prediction results based on the LLM.
#     dict
#         A dictionary containing detailed prediction results for each event pair,
#         including event IDs, event sentences, and parsed predictions.
#     """
#     # initialize the mention_map
#     dataset_folder = f"./datasets/{dataset}/"
#     mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
#     evt_mention_map = {
#         m_id: m for m_id, m in mention_map.items() if m["men_type"] == "evt"
#     }

#     results = lh(dataset, lh_threshold)
#     (tps, fps, tns, fns), (tps_trans, fps_trans, tns_trans, fns_trans) = results[
#         split_index_map[split]
#     ]
#     prediction_pairs = tps + fps

#     prediction_list, prediction_dict = llm_coref(
#         prediction_pairs,
#         evt_mention_map,
#         prompt=baseline_prompt,
#         parser=baseline_output_parser,
#     )

#     return prediction_list, prediction_dict


def llm_coref(
    event_pairs: List[Tuple[str, str]],
    mention_map: Dict,
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    gpt_version: str = "gpt-4",
    save_folder: str = "../../llm_results",
    text_key: str = "marked_doc",
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
    save_folder : str, optional
        Folder to save the prediction results. Default is "../../llm_results".


    Returns
    -------
    list of int
        A list of prediction results, where 1 represents 'True' and 0 represents 'False'.
    dict
        A dictionary containing detailed prediction results for each event pair,
        including event IDs, event sentences, and parsed predictions.
    """
    result_list = []
    result_dict = defaultdict(dict)

    # initialize the llm
    llm = ChatOpenAI(temperature=0.0, model=gpt_version)
    chain = LLMChain(llm=llm, prompt=prompt)

    # predict
    for evt_pair in tqdm(event_pairs):
        event1_id = evt_pair[0]
        event2_id = evt_pair[1]

        event1_data = mention_map.get(event1_id)
        event2_data = mention_map.get(event2_id)

        if event1_data is None or event2_data is None:
            continue

        event1 = get_context(event1_data, text_key)
        event2 = get_context(event2_data, text_key)

        predict = chain.run(event1=event1, event2=event2)
        predict_dict = parser.parse(predict)
        result_list.append(predict_dict["Answer"])

        format_prompt = chain.prompt.format_prompt(event1=event1, event2=event2)

        # format method to

        result_dict[evt_pair] = {
            "prompt": format_prompt,
            "event1_id": event1_id,
            "event2_id": event2_id,
            "event1": event1,
            "event2": event2,
            "predict_raw": predict,
            "predict_dict": predict_dict,
        }

    # Save the result_dict
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        save_folder, f"{gpt_version}_predict_result_{timestamp}.pkl"
    )
    pickle.dump(result_dict, open(save_path, "wb"))

    result_list = [1 if r == "True" else 0 for r in result_list]
    return result_list, result_dict

def get_biencoder_knn(
    dataset_folder: str,
    split: str,
    model_name: str,
    output_file: Path,
    ce_text_key: str = "marked_sentence",
    top_k: int = 10,
    device: str = "cuda",
    long: bool = False,
):
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    candidate_map = biencoder_nn(
        dataset_folder, split, model_name, long, top_k, device, text_key=ce_text_key
    )
    print(len(candidate_map))
    pickle.dump(candidate_map, open(output_file, "wb"))
    return candidate_map


@app.command()
def run_lh_llm_pipeline(
    dataset_folder: str,
    split: str,
    gpt_version: str = "gpt-4",
    save_folder: str = "../../llm_results",
    experiment_name: str = "baseline",
):
    """

    Parameters
    ----------
    dataset_folder
    split
    gpt_version
    template

    Returns
    -------

    """
    # set up openai api key
    _ = load_dotenv(find_dotenv())  # read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # read the mention_map from the dataset_folder
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))

    # Generate event pairs from the split and remove redundant cases using the 'Lemma Heuristic' method.
    mps, mps_trans = get_lh_pairs(mention_map, split, heu="lh", lh_threshold=0.05)
    tps, fps, tns, fns = mps
    event_pairs = tps + fps

    # debug
    event_pairs = event_pairs

    prompt, parser = prompt_and_parser_factory(experiment_name)

    result_list, _ = llm_coref(
        event_pairs,
        mention_map,
        prompt,
        parser,
        gpt_version,
        save_folder=save_folder,
    )

    # evaluate the result
    result_array = np.array(result_list)
    evaluate_result = evaluate(mention_map, event_pairs, similarity_matrix=result_array)

    return evaluate_result


if __name__ == "__main__":
    app()
