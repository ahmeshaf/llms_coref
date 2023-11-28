# Standard Library Imports
import os
import pickle
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

# Third-Party Library Imports
import numpy as np
import typer

# Local/Custom Imports
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser
from tqdm import tqdm

from coref_prompt_collections import baseline_output_parser, baseline_prompt
from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import b_cubed, ceafe, evaluate_documents, lea, muc
from helper import cluster, generate_key_file
from heuristic import lh

# global variables
split_index_map = {"train": 0, "dev": 1, "test": 2}

app = typer.Typer()


# helper functions
def read(key, response):
    return get_coref_infos("%s" % key, "%s" % response, False, False, True)


def run_llm_lh(
    dataset: str,
    split: str,
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    lh_threshold: float = 0.5,
) -> Tuple[List, Dict]:
    """
    Run the LLM (Language Model) with a linguistic heuristic (LH) for a specified dataset and split.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    split : str
        The data split (e.g., 'train', 'test', 'dev').
    prompt : PromptTemplate
        A template for the LLM prompt to query the model.
    parser : StructuredOutputParser
        A parser to extract structured output from the LLM's response.
    lh_threshold : float, optional
        The threshold value for the linguistic heuristic. Defaults to 0.5.

    Returns
    -------
    list
        A list of prediction results based on the LLM.
    dict
        A dictionary containing detailed prediction results for each event pair,
        including event IDs, event sentences, and parsed predictions.
    """
    # initialize the mention_map
    dataset_folder = f"./datasets/{dataset}/"
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    evt_mention_map = {
        m_id: m for m_id, m in mention_map.items() if m["men_type"] == "evt"
    }

    results = lh(dataset, lh_threshold)
    (tps, fps, tns, fns), (tps_trans, fps_trans, tns_trans, fns_trans) = results[
        split_index_map[split]
    ]
    prediction_pairs = tps + fps

    prediction_list, prediction_dict = llm_coref(
        prediction_pairs,
        evt_mention_map,
        prompt=baseline_prompt,
        parser=baseline_output_parser,
    )

    return prediction_list, prediction_dict


def llm_coref(
    prediction_pairs: List[Tuple[str, str]],
    mention_map: Dict,
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    gpt_version: str = "gpt-4",
    save_folder: str = "../../llm_results",
) -> Dict:
    """
    Predict coreference using the LLM (Language Model) for provided event pairs.

    Parameters
    ----------
    prediction_pairs : list of tuple of str
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
    result_dict = defaultdict(dict)
    """
    result_list = []
    result_dict = defaultdict(dict)

    # initialize the llm
    llm = ChatOpenAI(temperature=0.0, model=gpt_version)
    chain = LLMChain(llm=llm, prompt=prompt)

    # predict
    for evt_pair in tqdm(prediction_pairs):
        event1_id = evt_pair[0]
        event2_id = evt_pair[1]

        event1_data = mention_map.get(event1_id)
        event2_data = mention_map.get(event2_id)

        if event1_data is None or event2_data is None:
            continue

        event1 = event1_data["bert_sentence"]
        event2 = event2_data["bert_sentence"]

        predict = chain.run(event1=event1, event2=event2)
        predict_dict = parser.parse(predict)
        result_list.append(predict_dict["Answer"])

        result_dict[evt_pair] = {
            "event1_id": event1_id,
            "event2_id": event2_id,
            "event1": event1,
            "event2": event2,
            "predict_dict": predict_dict,
        }

    # save the result_dict
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "llm_predict_result.pkl")
    pickle.dump(result_dict, open(save_path, "wb"))

    # modify the reuslt_list
    result_list = [1 if r == "True" else 0 for r in result_list]
    return result_list, result_dict


def evaluate(
    mention_map: Dict[str, Dict[str, str]],
    prediction_pairs: List[Tuple[str, str]],
    similarity_matrix: np.ndarray,
    tmp_folder: str = "../../tmp/",
) -> Dict[str, Tuple[float, float, float]]:
    """
    Evaluate the prediction results using various coreference resolution metrics.

    Parameters
    ----------
    mention_map : dict
        A mapping of mentions to their attributes. Each attribute should have a 'gold_cluster' key.
    prediction_pairs : list of tuple
        List of tuples representing predicted pairs of mentions.
    similarity_matrix : np.ndarray
        A one-dimensional array representing the predicted results for each pair of mentions.
    tmp_folder : str, optional
        Directory path to store temporary files. Defaults to '../../tmp/'.

    Returns
    -------
    dict
        A dictionary containing evaluation results for various coreference metrics. The keys include:
        - 'MUC': (recall, precision, f-score)
        - 'B-Cubed': (recall, precision, f-score)
        - 'CEAF-E': (recall, precision, f-score)
        - 'LEA': (recall, precision, f-score)
    """
    # Create the key file with gold clusters from mention map
    curr_mentions = sorted(mention_map.keys())
    curr_gold_cluster_map = [
        (men, mention_map[men]["gold_cluster"]) for men in curr_mentions
    ]
    gold_key_file = tmp_folder + "/gold_clusters.keyfile"
    generate_key_file(curr_gold_cluster_map, "evt", tmp_folder, gold_key_file)

    # Run clustering using prediction_pairs and similarity_matrix
    mid2cluster = cluster(curr_mentions, prediction_pairs, similarity_matrix)

    # Create a predictions key file
    system_key_file = tmp_folder + "/predicted_clusters.keyfile"
    generate_key_file(mid2cluster.items(), "evt", tmp_folder, system_key_file)

    # Evaluation on gold and prediction key files.
    doc = read(gold_key_file, system_key_file)

    mr, mp, mf = np.round(np.round(evaluate_documents(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate_documents(doc, b_cubed), 3) * 100, 1)
    cr, cp, cf = np.round(np.round(evaluate_documents(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate_documents(doc, lea), 3) * 100, 1)

    results = {
        "MUC": (mr, mp, mf),
        "B-Cubed": (br, bp, bf),
        "CEAF-E": (cr, cp, cf),
        "LEA": (lr, lp, lf),
    }

    return results


@app.command()
def run_llm_pipeline(dataset, split, gpt_version, template):
    pass


if __name__ == "__main__":
    app()
