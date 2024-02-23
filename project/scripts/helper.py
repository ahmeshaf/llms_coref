import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from coval.conll.reader import get_coref_infos
from coval.eval.evaluator import b_cubed, ceafe, evaluate_documents, lea, muc

TRAIN = "train"
DEV = "dev"
TEST = "test"


def cluster_cc(affinity_matrix, threshold=0.8):
    """
    Find connected components using the affinity matrix and threshold -> adjacency matrix
    Parameters
    ----------
    affinity_matrix: np.array
    threshold: float

    Returns
    -------
    list, np.array
    """
    adjacency_matrix = csr_matrix(affinity_matrix > threshold)
    clusters, labels = connected_components(
        adjacency_matrix, return_labels=True, directed=False
    )
    return clusters, labels


def remove_puncts(target_str):
    return target_str
    # return target_str.translate(str.maketrans('', '', string.punctuation)).lower()


def jc(arr1, arr2):
    return len(set.intersection(set(arr1), set(arr2))) / len(set.union(set(arr1), set(arr2)))


def generate_mention_pairs(mention_map, split):
    """

    Parameters
    ----------
    mention_map: dict
    split: str (train/dev/test)

    Returns
    -------
    list: A list of all possible mention pairs within a topic
    """
    split_mention_ids = sorted(
        [m_id for m_id, m in mention_map.items() if m["split"] == split]
    )
    topic2mentions = {}
    for m_id in split_mention_ids:
        try:
            topic = mention_map[m_id][
                "predicted_topic"
            ]  # specifically for the test set of ECB
        except KeyError:
            # print("HELLO")
            topic = None
        if not topic:
            topic = mention_map[m_id]["topic"]
        if topic not in topic2mentions:
            topic2mentions[topic] = []
        topic2mentions[topic].append(m_id)

    mention_pairs = []

    for mentions in topic2mentions.values():
        list_mentions = list(mentions)
        for i in range(len(list_mentions)):
            for j in range(i + 1):
                m1 = list_mentions[i]
                m2 = list_mentions[j]
                doc_sent_id1 = (mention_map[m1]["doc_id"], mention_map[m1]["sentence_id"])
                doc_sent_id2 = (mention_map[m2]["doc_id"], mention_map[m2]["sentence_id"])
                if m1 > m2:
                    m2, m1 = m1, m2
                if i != j and mention_map and doc_sent_id1 != doc_sent_id2:
                    mention_pairs.append((m1, m2))
    # print(len(mention_pairs))
    # mention_pairs = list(set(mention_pairs))
    # print(len(mention_pairs))
    return mention_pairs


def generate_key_file(coref_map_tuples, name, out_dir, out_file_path):
    """

    Parameters
    ----------
    coref_map_tuples: list
    name: str
    out_dir: str
    out_file_path: str

    Returns
    -------
    None
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    clus_to_int = {}
    clus_number = 0
    with open(out_file_path, "w") as of:
        of.write("#begin document (%s);\n" % name)
        for i, map_ in enumerate(coref_map_tuples):
            en_id = map_[0]
            clus_id = map_[1]
            if clus_id in clus_to_int:
                clus_int = clus_to_int[clus_id]
            else:
                clus_to_int[clus_id] = clus_number
                clus_number += 1
                clus_int = clus_to_int[clus_id]
            of.write("%s\t0\t%d\t%s\t(%d)\n" % (name, i, en_id, clus_int))
        of.write("#end document\n")


def cluster(mentions, mention_pairs, similarities, threshold=0.0):
    m_id2ind = {m: i for i, m in enumerate(mentions)}

    mention_ind_pairs = [(m_id2ind[mp[0]], m_id2ind[mp[1]]) for mp in mention_pairs]
    rows, cols = zip(*mention_ind_pairs)

    # create similarity matrix from the similarities
    n = len(mentions)
    similarity_matrix = np.identity(n)
    similarity_matrix[rows, cols] = similarities

    clusters, labels = cluster_cc(similarity_matrix, threshold=threshold)
    m_id2cluster = [(m, i) for m, i in zip(mentions, labels)]
    return m_id2cluster


def cluster_greedy(mentions, mention_pairs, similarities, threshold=0.5):
    m_id2ind = {m: i for i, m in enumerate(mentions)}

    mention_ind_pairs = [(m_id2ind[mp[0]], m_id2ind[mp[1]]) for mp in mention_pairs]
    rows, cols = zip(*mention_ind_pairs)

    mention_pairs_sorted = sorted(
        [(m1, m2, score) for (m1, m2), score in zip(mention_pairs, similarities)],
        key=lambda x: x[-1],
        reverse=True,
    )

    # create similarity matrix from the similarities
    n = len(mentions)
    similarity_matrix = np.identity(n)
    similarity_matrix[rows, cols] = similarities
    similarity_matrix[cols, rows] = similarities

    base_clusters = {i: [i] for i in range(len(mentions))}

    for m1, m2, score in mention_pairs_sorted:
        m1_ind = m_id2ind[m1]
        m2_ind = m_id2ind[m2]

        if m1_ind in base_clusters and m2_ind in base_clusters:
            c_1 = base_clusters[m1_ind]
            c_2 = base_clusters[m2_ind]
            # if score > threshold:

            while not isinstance(c_2, list):
                c_2 = base_clusters[c_2]
            while not isinstance(c_1, list):
                c_1 = base_clusters[c_1]
            score = np.mean([similarity_matrix[i, j] for i in c_1 for j in c_2])
            if score > threshold:
                for m_ in c_2:
                    if m_ not in c_1:
                        c_1.append(m_)
                    base_clusters[m2_ind] = m1_ind
    clusters = [clus for clus in base_clusters.values() if isinstance(clus, list)]
    # clusters, labels = cluster_cc(similarity_matrix, threshold=threshold)
    m_id2cluster = {m: i for i, ms in enumerate(clusters) for m in ms}
    return [(m, m_id2cluster[m_id2ind[m]]) for m in list(mentions)]
    # return m_id2cluster


def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    return sum(predicted_labels == true_labels) / len(predicted_labels)


def precision(predicted_labels, true_labels):
    """
    Precision is True Positives / All Positives Predictions
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(predicted_labels)


def recall(predicted_labels, true_labels):
    """
    Recall is True Positives / All Positive Labels
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(true_labels)


def f1_score(predicted_labels, true_labels):
    """
    F1 score is the harmonic mean of precision and recall
    """
    P = precision(predicted_labels, true_labels)
    R = recall(predicted_labels, true_labels)
    return 2 * P * R / (P + R)


def ensure_path(file: Path):
    # Ensure the file is Path object
    if not isinstance(file, Path):
        file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir(parents=True)


def ensure_dir(dir_path: Path):
    # Ensure the file is Path object
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def read(key, response):
    return get_coref_infos("%s" % key, "%s" % response, False, False, True)


def evaluate(
    mention_map: Dict[str, Dict[str, str]],
    split_mention_ids: List[str],
    prediction_pairs: List[Tuple[str, str]],
    similarity_matrix: np.ndarray,
    tmp_folder: str = "/tmp/",
    threshold: float = 0.5,
    greedy: bool = False
) -> Dict[str, Tuple[float, float, float]]:
    """
    Evaluate the prediction results using various coreference resolution metrics.

    Parameters
    ----------
    threshold: float
    mention_map : dict
        A mapping of mentions to their attributes. Each attribute should have a 'gold_cluster' key.
    split_mention_ids: List[str]
        mention ids of current split
    prediction_pairs : list of tuple
        List of tuples representing predicted pairs of mentions.
    similarity_matrix : np.ndarray
        A one-dimensional array representing the predicted results for each pair of mentions.
    tmp_folder : str, optional
        Directory path to store temporary files. Defaults to '../../tmp/'.
    greedy : bool

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
    split_mention_ids = sorted(list(split_mention_ids))
    curr_gold_cluster_map = [
        (men, mention_map[men]["gold_cluster"]) for men in split_mention_ids
    ]
    gold_key_file = tmp_folder + "/gold_clusters.keyfile"
    generate_key_file(curr_gold_cluster_map, "evt", tmp_folder, gold_key_file)

    # Run clustering using prediction_pairs and similarity_matrix
    if greedy:
        mid2cluster = cluster_greedy(split_mention_ids, prediction_pairs, similarity_matrix)
    else:
        mid2cluster = cluster(split_mention_ids, prediction_pairs, similarity_matrix, threshold=threshold)

    # Create a predictions key file
    system_key_file = tmp_folder + "/predicted_clusters.keyfile"
    generate_key_file(mid2cluster, "evt", tmp_folder, system_key_file)
    # Evaluation on gold and prediction key files.
    doc = read(gold_key_file, system_key_file)

    mr, mp, mf = np.round(np.round(evaluate_documents(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate_documents(doc, b_cubed), 3) * 100, 1)
    cr, cp, cf = np.round(np.round(evaluate_documents(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate_documents(doc, lea), 3) * 100, 1)

    conll = ( mf + bf + cf )/3

    results = {
        "MUC": (mr, mp, mf),
        "B-Cubed": (br, bp, bf),
        "CEAF-E": (cr, cp, cf),
        "CONLL": (mf + bf + cf) / 3,
        "LEA": (lr, lp, lf),
    }

    print(f"{br} & {bp} & {bf} & {conll}")

    return results


def get_split_ids(mention_map, split):
    return [
        m_id
        for m_id, men in mention_map.items()
        if men["men_type"] == "evt" and men["split"] == split
    ]
    
def filter_topic_aligned_cross_doc_pairs(mention_pairs: List, mention_map: Dict):
    """
    Filters mention pairs to retain only those that are in the same topic but in different documents and sentences.
    """
    return [
        (m1, m2)
        for m1, m2 in mention_pairs
        if mention_map[m1]["topic"] == mention_map[m2]["topic"] and
           (mention_map[m1]["doc_id"], mention_map[m1]["sentence_id"]) !=
           (mention_map[m2]["doc_id"], mention_map[m2]["sentence_id"])
    ]
    
