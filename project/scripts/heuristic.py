import os.path
import pickle
import typer

from pathlib import Path
from tqdm import tqdm

from .helper import *

app = typer.Typer()


def get_mention_pair_similarity_lemma2(
    mention_pairs, mention_map, relations, threshold=0.05
):
    """
    Generate the similarities for mention pairs

    Parameters
    ----------
    mention_pairs: list
    mention_map: dict
    relations: list
        The list of relations represented as a triple: (head, label, tail)
    working_folder: str

    Returns
    -------
    list
    """
    similarities = []

    doc_ids = []
    doc2id = {doc: i for i, doc in enumerate(doc_ids)}

    # generate similarity using the mention text
    for pair in tqdm(mention_pairs, desc="Generating Similarities"):
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = men_map1["mention_text"].lower()
        men_text2 = men_map2["mention_text"].lower()

        def jc(arr1, arr2):
            return len(set.intersection(arr1, arr2)) / len(set.union(arr1, arr2))
            # return len(set.intersection(arr1, arr2))

        doc_id1 = men_map1["doc_id"]
        sentence_tokens1 = [tok for tok in men_map1["sentence_tokens"]]

        sentence_tokens2 = [tok for tok in men_map2["sentence_tokens"]]

        sent_sim = jc(set(sentence_tokens1), set(sentence_tokens2))
        lemma_sim = float(
            men_map1["lemma"].lower() in men_text2
            or men_map2["lemma"].lower() in men_text1
            or men_map1["lemma"].lower() in men_map2["lemma"].lower()
        )

        lemma1 = men_map1["lemma"].lower()
        lemma2 = men_map2["lemma"].lower()
        if lemma1 > lemma2:
            pair_tuple = (lemma2, lemma1)
        else:
            pair_tuple = (lemma1, lemma2)

        similarities.append(
            (lemma_sim or pair_tuple in relations) and sent_sim > threshold
        )

    return np.array(similarities)


def get_mention_pair_similarity_lemma(
    mention_pairs, mention_map, syn_lemma_pairs, threshold=0.05
):
    similarities = []

    # generate similarity using the mention text
    for pair in tqdm(mention_pairs, desc="Generating Similarities"):
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = remove_puncts(men_map1["mention_text"].lower())
        men_text2 = remove_puncts(men_map2["mention_text"].lower())
        lemma1 = remove_puncts(men_map1["lemma"].lower())
        lemma2 = remove_puncts(men_map2["lemma"].lower())

        sentence_tokens1 = [tok.lower() for tok in men_map1["sentence_tokens"]]

        sentence_tokens2 = [tok.lower() for tok in men_map2["sentence_tokens"]]

        sent_sim = jc(set(sentence_tokens1), set(sentence_tokens2))

        lemma_sim = float(
            lemma1 in men_text2 or lemma2 in men_text1 or men_text1 in lemma2
        )
        pair_tuple = tuple(sorted([lemma1, lemma2]))

        similarities.append(
            (lemma_sim or pair_tuple in syn_lemma_pairs) and sent_sim > threshold
        )

    return np.array(similarities)


def get_all_mention_pairs_labels_split(mention_map, split):
    split_mention_pairs = generate_mention_pairs(mention_map, split)
    split_labels = [
        int(mention_map[m1]["gold_cluster"] == mention_map[m2]["gold_cluster"])
        for m1, m2 in split_mention_pairs
    ]
    split_pairs_labels = list(zip(split_mention_pairs, split_labels))
    return split_pairs_labels


def get_all_mention_pairs_labels(mention_map):
    all_mention_pairs_labels = []
    for split in [TRAIN, DEV, TEST]:
        split_pairs_labels = get_all_mention_pairs_labels_split(mention_map, split)
        all_mention_pairs_labels.append(split_pairs_labels)
    return all_mention_pairs_labels


def get_lemma_pairs_labels(mention_map, pairs_labels):
    lemma_pairs_labels = []
    for (m1, m2), label in pairs_labels:
        lemma1 = remove_puncts(mention_map[m1]["lemma"].lower())
        lemma2 = remove_puncts(mention_map[m2]["lemma"].lower())
        if lemma1 > lemma2:
            pair_tuple = (lemma2, lemma1)
        else:
            pair_tuple = (lemma1, lemma2)

        lemma_pairs_labels.append((pair_tuple, label))
    return lemma_pairs_labels


def generate_tp_fp_tn_fn(
    mention_pairs,
    ground_truth,
    mention_map,
    syn_lemma_pairs,
    threshold=0.05,
    doc_sent_map=None,
):
    similarities = get_mention_pair_similarity_lemma2(
        mention_pairs, mention_map, syn_lemma_pairs, threshold=threshold
    )

    lemma_coref = similarities > threshold
    # print('all positives:', lemma_coref.sum())

    tps = np.logical_and(lemma_coref, ground_truth).nonzero()
    tps = [mention_pairs[i] for i in tps[0]]
    fps = np.logical_and(lemma_coref, np.logical_not(ground_truth)).nonzero()
    fps = [mention_pairs[i] for i in fps[0]]
    tns = np.logical_and(
        np.logical_not(lemma_coref), np.logical_not(ground_truth)
    ).nonzero()
    tns = [mention_pairs[i] for i in tns[0]]
    fns = np.logical_and(np.logical_not(lemma_coref), ground_truth).nonzero()
    fns = [mention_pairs[i] for i in fns[0]]

    print("true positives:", len(tps))
    print("false positives:", len(fps))
    print("true negatives:", len(tns))
    print("false negatives:", len(fns))

    ind2m_id = list(mention_map.keys())
    n = len(ind2m_id)
    m_id2ind = {m: i for i, m in enumerate(ind2m_id)}
    sim_matrix = np.zeros((n, n))
    for (m1, m2), sim in zip(mention_pairs, similarities):
        sim_matrix[m_id2ind[m1], m_id2ind[m2]] = sim
    clusters, labels = cluster_cc(sim_matrix, threshold=0.15)
    m_id2cluster = {m: i for m, i in zip(ind2m_id, labels)}
    lemma_coref_transitive = np.array(
        [m_id2cluster[m1] == m_id2cluster[m2] for m1, m2 in mention_pairs]
    )

    tps_trans = np.logical_and(lemma_coref_transitive, ground_truth).nonzero()
    tps_trans = [mention_pairs[i] for i in tps_trans[0]]
    fps_trans = np.logical_and(
        lemma_coref_transitive, np.logical_not(ground_truth)
    ).nonzero()
    fps_trans = [mention_pairs[i] for i in fps_trans[0]]
    tns_trans = np.logical_and(
        np.logical_not(lemma_coref_transitive), np.logical_not(ground_truth)
    ).nonzero()
    tns_trans = [mention_pairs[i] for i in tns_trans[0]]
    fns_trans = np.logical_and(
        np.logical_not(lemma_coref_transitive), ground_truth
    ).nonzero()
    fns_trans = [mention_pairs[i] for i in fns_trans[0]]

    print("\nAfter transitive closure\ntrue positives:", len(tps_trans))
    print("false positives:", len(fps_trans))
    print("true negatives:", len(tns_trans))
    print("false negatives:", len(fns_trans))
    return (tps, fps, tns, fns), (tps_trans, fps_trans, tns_trans, fns_trans)


def lh(dataset, threshold=0.05):
    """

    Parameters
    ----------
    dataset: str
        The dataset name: ecb/gvc
    threshold: double

    Returns
    -------
    None: Save the predicted mention pairs from the dataset in the dataset's folder
        Directory location: ./datasets/dataset/lh/
    """
    dataset_folder = f"../../datasets/{dataset}/"
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    evt_mention_map = {
        m_id: m for m_id, m in mention_map.items() if m["men_type"] == "evt"
    }
    (
        tr_mention_pairs_labels,
        dev_mention_pairs_labels,
        test_mention_pairs_labels,
    ) = get_all_mention_pairs_labels(evt_mention_map)

    train_lemma_pairs_labels = get_lemma_pairs_labels(
        evt_mention_map, tr_mention_pairs_labels
    )

    train_syn_lemma_pairs = set([p for p, l in train_lemma_pairs_labels if l == 1])
    train_non_syn_pairs = set(
        [
            p
            for p, l in train_lemma_pairs_labels
            if l == 0 and p not in train_syn_lemma_pairs
        ]
    )

    # train_syn_lemma_pls = [(p, l) for p, l in train_lemma_pairs_labels if p in train_syn_lemma_pairs]
    # train_non_syn_lps = [(p, l) for p, l in train_lemma_pairs_labels if p in train_non_syn_pairs]
    result = []
    for split, pair_labels in zip(
        [TRAIN, DEV, TEST],
        [tr_mention_pairs_labels, dev_mention_pairs_labels, test_mention_pairs_labels],
    ):
        print(split)
        pairs, labels = zip(*pair_labels)
        (mps, mps_trans) = generate_tp_fp_tn_fn(
            pairs,
            np.array(labels),
            mention_map,
            train_syn_lemma_pairs,
            threshold=threshold,
        )
        # pickle.dump((mps, mps_trans), open(f'./datasets/{dataset}/lh/mp_mp_t_{split}.pkl', 'wb'))
        result.append((mps, mps_trans))
    return result


def lh_oracle(dataset, threshold=0.05):
    """

    Parameters
    ----------
    dataset: str
        The dataset name: ecb/gvc
    threshold: double

    Returns
    -------
    None: Save the predicted mention pairs from the dataset in the dataset's folder
        Directory location: ./datasets/dataset/lh_oracle/
    """
    dataset_folder = f"./datasets/{dataset}/"
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    evt_mention_map = {
        m_id: m for m_id, m in mention_map.items() if m["men_type"] == "evt"
    }
    (
        tr_mention_pairs_labels,
        dev_mention_pairs_labels,
        test_mention_pairs_labels,
    ) = get_all_mention_pairs_labels(evt_mention_map)

    train_syn_lemma_pairs = get_lemma_pairs_labels(
        evt_mention_map, tr_mention_pairs_labels
    )
    dev_syn_lemma_pairs = get_lemma_pairs_labels(
        evt_mention_map, dev_mention_pairs_labels
    )
    test_syn_lemma_pairs = get_lemma_pairs_labels(
        evt_mention_map, test_mention_pairs_labels
    )

    tr_syn_lemma_pairs = set([p for p, l in train_syn_lemma_pairs if l == 1])
    dev_syn_lemma_pairs = set([p for p, l in dev_syn_lemma_pairs if l == 1])
    test_syn_lemma_pairs = set([p for p, l in test_syn_lemma_pairs if l == 1])

    split_syn_lemma = {
        split: syns
        for split, syns in zip(
            [TRAIN, DEV, TEST],
            [tr_syn_lemma_pairs, dev_syn_lemma_pairs, test_syn_lemma_pairs],
        )
    }

    all_syn_lemmas = tr_syn_lemma_pairs.union(dev_syn_lemma_pairs).union(
        test_syn_lemma_pairs
    )

    pass
    for split, pair_labels in zip(
        [TRAIN, DEV, TEST],
        [tr_mention_pairs_labels, dev_mention_pairs_labels, test_mention_pairs_labels],
    ):
        print("-------", split, "--------")
        pairs, labels = zip(*pair_labels)
        (mps, mps_trans) = generate_tp_fp_tn_fn(
            pairs,
            np.array(labels),
            mention_map,
            split_syn_lemma[split],
            threshold=threshold,
        )
        pickle.dump(
            (mps, mps_trans),
            open(f"./datasets/{dataset}/lh_oracle/mp_mp_t_{split}.pkl", "wb"),
        )


def get_lh_pairs(mention_map, split, heu="lh", lh_threshold=0.15, lemma_pairs=None):
    evt_mention_map = {
        m_id: m for m_id, m in mention_map.items() if m["men_type"] == "evt"
    }
    split_mention_pairs_labels = get_all_mention_pairs_labels_split(
        evt_mention_map, split
    )

    if heu == "lh":
        train_mention_pairs_labels = get_all_mention_pairs_labels_split(
            evt_mention_map, "train"
        )
        train_syn_lemma_pairs = get_lemma_pairs_labels(
            evt_mention_map, train_mention_pairs_labels
        )
        split_syn_lemma_pairs = set([p for p, l in train_syn_lemma_pairs if l == 1])
    else:
        split_syn_lemma_pairs = get_lemma_pairs_labels(
            evt_mention_map, split_mention_pairs_labels
        )
        split_syn_lemma_pairs = set([p for p, l in split_syn_lemma_pairs if l == 1])

    if lemma_pairs is not None:
        split_syn_lemma_pairs = lemma_pairs

    pairs, labels = zip(*split_mention_pairs_labels)
    (m_pairs, m_pairs_trans) = generate_tp_fp_tn_fn(
        pairs,
        np.array(labels),
        mention_map,
        split_syn_lemma_pairs,
        threshold=lh_threshold,
    )
    return m_pairs, m_pairs_trans


@app.command()
def save_lh_pairs(
    dataset_folder: str, split: str, heu: str, lh_threshold: float, pairs_out_file: Path
):
    ensure_path(pairs_out_file)
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    m_pairs, _ = get_lh_pairs(mention_map, split, heu=heu, lh_threshold=lh_threshold)

    # use only the positive predictions
    tp_fp = m_pairs[0] + m_pairs[1]
    mention_pairs = set()
    for m1, m2 in tp_fp:
        p = tuple(sorted((m1, m2)))
        mention_pairs.add(p)
    print(len(mention_pairs))
    pickle.dump(mention_pairs, open(pairs_out_file, "wb"))


if __name__ == "__main__":
    app()
