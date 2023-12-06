import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import typer
from tqdm import tqdm

from .bert_pipeline import ensure_path

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer()


# helpers
def add_context_to_mentions(
    mention_map: Dict[str, Dict[str, str]],
    doc_sent_map: Dict[str, Dict[str, Dict[str, str]]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Enhances each mention in the mention map with context from the document-sentence map.

    Parameters:
    - mention_map (Dict[str, Dict[str, str]]): A map of mentions.
    - doc_sent_map (Dict[str, Dict[str, Dict[str, str]]]): A map from document IDs to sentences.

    Returns:
    - Dict[str, Dict[str, List[str]]]: The mention map with added left and right context for each mention.
    """
    augmented_mention_map = mention_map.copy()

    for key, value in tqdm(mention_map.items()):
        doc_id = value["doc_id"]
        sent_id = int(value["sentence_id"])

        doc_unit = doc_sent_map.get(doc_id, None)
        if doc_unit is None:
            print(f"doc_id {doc_id} not found")
            continue

        index = doc_unit.keys()
        sentence_list = [doc_unit.get(str(i))["sentence"] for i in index]
        if len(sentence_list) != len(index):
            raise ValueError(
                f"Mismatch in number of sentences for document ID {doc_id}"
            )

        context_left = sentence_list[:sent_id]
        context_right = sentence_list[sent_id + 1 :]

        if len(context_left) + len(context_right) + 1 != len(sentence_list):
            raise ValueError(f"Context lengths do not match for document ID {doc_id}")

        augmented_mention_map[key]["context_left"] = context_left
        augmented_mention_map[key]["context_right"] = context_right

    return augmented_mention_map


def group_mentions_by_label(
    mention_map: Dict[str, Dict[str, any]]
) -> Dict[str, List[Dict[str, any]]]:
    """
    Groups mentions in the mention map by their labels.

    Parameters:
    - mention_map (Dict[str, Dict[str, any]]): A dictionary of mentions.

    Returns:
    - Dict[str, List[Dict[str, any]]]: A dictionary grouping mentions by their labels.
    """
    label_group = defaultdict(list)
    for _, value in mention_map.items():  # Removed tqdm for simplicity
        label = value.get("gold_cluster")
        label_group[label].append(value)

    return label_group


def add_candidates_and_count_singletons(
    mention_map: Dict[str, Dict[str, str]], label_group: Dict[str, List[Dict[str, str]]]
) -> Tuple[Dict[str, Dict[str, str]], int]:
    """
    Adds positive candidate mentions to each entry in the mention map and counts singletons.

    Parameters:
    - mention_map (Dict[str, Dict[str, str]]): A map of mentions.
    - label_group (Dict[str, List[Dict[str, str]]]): Grouped mentions by labels.

    Returns:
    - Tuple[Dict[str, Dict[str, str]], int]: The updated mention map and the count of singletons.
    """
    singleton_count = 0
    updated_mention_map = {}  # Creating a new dictionary

    for key, value in mention_map.items():
        label = value.get("gold_cluster")
        if label is not None:
            positive_candidates = label_group.get(label, None)
            if positive_candidates and len(positive_candidates) == 1:
                logging.info(
                    f"Singleton Found! label: {label}, id: {key}, sentence: {value['sentence']}, "
                    f"mention_text: {value['mention_text']}, type: {value['type']}"
                )
                singleton_count += 1

            updated_mention_map[key] = value.copy()
            updated_mention_map[key]["positive_candidates"] = positive_candidates

    logging.info(f"Total number of singletons: {singleton_count}")
    return updated_mention_map, singleton_count


@app.command()
def augment_context(
    mention_map_path,
    doc_mention_map_path,
    save_path="../corpus/mention_map_with_context.pkl",
):
    mention_map = pickle.load(open(mention_map_path, "rb"))
    doc_sent_map = pickle.load(open(doc_mention_map_path, "rb"))
    augmented_mention_map = add_context_to_mentions(mention_map, doc_sent_map)
    pickle.dump(augmented_mention_map, open(save_path, "wb"))


@app.command()
def augment_candidates(
    mention_map_path,
    map_save_path="../corpus/mention_map_with_candidates.pkl",
    label_group_save_path="../corpus/label_group.pkl",
):
    mention_map = pickle.load(open(mention_map_path, "rb"))
    label_group = group_mentions_by_label(mention_map)
    pickle.dump(label_group, open(label_group_save_path, "wb"))
    updated_mention_map, _ = add_candidates_and_count_singletons(
        mention_map, label_group
    )
    pickle.dump(updated_mention_map, open(map_save_path, "wb"))


@app.command()
def add_debug_split(mention_map_path: Path, aug_mention_map_path: Path):
    ensure_path(aug_mention_map_path)
    mention_map = pickle.load(open(mention_map_path, "rb"))

    debug_docs = defaultdict(set)
    for m_id, mention in mention_map.items():
        if mention["men_type"] == "evt" and mention["split"] == "dev":
            topic_id = mention["topic"]
            doc_id = mention["doc_id"]
            debug_docs[topic_id].add(doc_id)
    random.seed(42)
    debug_docs_list = set()
    for doc_list in debug_docs.values():
        docs = random.sample(doc_list, k=5)
        debug_docs_list.update(docs)

    for m_id, mention in mention_map.items():
        doc_id = mention["doc_id"]
        if doc_id in debug_docs_list:
            mention["split"] = "debug_split"

    pickle.dump(mention_map, open(aug_mention_map_path, "wb"))


if __name__ == "__main__":
    app()
