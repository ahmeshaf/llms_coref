import os.path
import pickle
import typer

from datasets import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from .helper import *
from .nn_method.bi_encoder import BiEncoder
from .nn_method.helper import (
    tokenize_bi,
    get_arg_attention_mask_wrapper,
    create_faiss_db,
    VectorDatabase,
    process_batch,
)

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

    within_doc_similarities = []

    # doc_sent_map = pickle.load(open(working_folder + '/doc_sent_map.pkl', 'rb'))
    # doc_sims = pickle.load(open(working_folder + '/doc_sims_path.pkl', 'rb'))
    doc_ids = []

    # for doc_id, _ in list(doc_sent_map.items()):
    #     doc_ids.append(doc_id)

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
        # sent_id1 = int(men_map1['sentence_id'])
        # all_sent_ids1 = {str(sent_id1 - 1), str(sent_id1), str(sent_id1 + 1)}
        # all_sent_ids1 = {str(sent_id1)}
        #
        # doc_id2 = men_map2['doc_id']
        # sent_id2 = int(men_map2['sentence_id'])
        # all_sent_ids2 = {str(sent_id2 - 1), str(sent_id2), str(sent_id2 + 1)}
        #
        # all_sent_ids2 = {str(sent_id2)}

        # sentence_tokens1 = [tok for sent_id in all_sent_ids1 if sent_id in doc_sent_map[doc_id1]
        #                     for tok in doc_sent_map[doc_id1][sent_id]['sentence_tokens']]
        #
        # sentence_tokens2 = [tok for sent_id in all_sent_ids2 if sent_id in doc_sent_map[doc_id2]
        #                     for tok in doc_sent_map[doc_id2][sent_id]['sentence_tokens']]

        sentence_tokens1 = [tok for tok in men_map1["sentence_tokens"]]

        sentence_tokens2 = [tok for tok in men_map2["sentence_tokens"]]

        sent_sim = jc(set(sentence_tokens1), set(sentence_tokens2))
        # sent_sim = jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens']))
        # doc_sim = doc_sims[doc2id[men_map1['doc_id']], doc2id[men_map2['doc_id']]]
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

        # similarities.append((lemma_sim or pair_tuple in relations))
        similarities.append(
            (lemma_sim or pair_tuple in relations) and sent_sim > threshold
        )
        # similarities.append((lemma_sim) and sent_sim > 0.05)
        # similarities.append((lemma_sim + 0.3*sent_sim)/2)

    return np.array(similarities)


def get_mention_pair_similarity_lemma(
    mention_pairs, mention_map, syn_lemma_pairs, threshold=0.05, doc_sent_map=None
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

        # doc_id1 = men_map1['doc_id']
        # sent_id1 = int(men_map1['sentence_id'])
        # all_sent_ids1 = {str(sent_id1 - 1), str(sent_id1), str(sent_id1 + 1)}
        # all_sent_ids1 = {str(sent_id1)}
        #
        # doc_id2 = men_map2['doc_id']
        # sent_id2 = int(men_map2['sentence_id'])
        # all_sent_ids2 = {str(sent_id2 - 1), str(sent_id2), str(sent_id2 + 1)}
        #
        # all_sent_ids2 = {str(sent_id2)}

        # sentence_tokens1 = [tok for sent_id in all_sent_ids1 if sent_id in doc_sent_map[doc_id1]
        #                     for tok in doc_sent_map[doc_id1][sent_id]['sentence_tokens']]
        #
        # sentence_tokens2 = [tok for sent_id in all_sent_ids2 if sent_id in doc_sent_map[doc_id2]
        #                     for tok in doc_sent_map[doc_id2][sent_id]['sentence_tokens']]

        sentence_tokens1 = [tok.lower() for tok in men_map1["sentence_tokens"]]

        sentence_tokens2 = [tok.lower() for tok in men_map2["sentence_tokens"]]

        sent_sim = jc(set(sentence_tokens1), set(sentence_tokens2))
        # sent_sim = jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens']))
        # doc_sim = doc_sims[doc2id[men_map1['doc_id']], doc2id[men_map2['doc_id']]]
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

        # lemma_pair = tuple(sorted([remove_puncts(mention_map[m1]['lemma'].lower()),
        #                            remove_puncts(mention_map[m2]['lemma'].lower())]))
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

    lemma_coref = similarities > 0.15
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


def lh_split(heu, dataset, split, threshold=0.05):
    dataset_folder = f"./datasets/{dataset}/"
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    evt_mention_map = {
        m_id: m for m_id, m in mention_map.items() if m["men_type"] == "evt"
    }
    split_mention_pairs_labels = get_all_mention_pairs_labels_split(
        evt_mention_map, split
    )

    if heu == "lh":
        train_menrion_pairs_labels = get_all_mention_pairs_labels_split(
            evt_mention_map, "train"
        )
        train_syn_lemma_pairs = get_lemma_pairs_labels(
            evt_mention_map, train_menrion_pairs_labels
        )
        split_syn_lemma_pairs = set([p for p, l in train_syn_lemma_pairs if l == 1])
    else:
        split_syn_lemma_pairs = get_lemma_pairs_labels(
            evt_mention_map, split_mention_pairs_labels
        )
        split_syn_lemma_pairs = set([p for p, l in split_syn_lemma_pairs if l == 1])

    pairs, labels = zip(*split_mention_pairs_labels)
    (mps, mps_trans) = generate_tp_fp_tn_fn(
        pairs, np.array(labels), mention_map, split_syn_lemma_pairs, threshold=threshold
    )
    return mps, mps_trans


def load_biencoder(model_path, long=False, linear_weights_path=None):
    """

    Parameters
    ----------
    model_path: str
    long: bool
    linear_weights_path: str

    Returns
    -------
    BiEncoder
    """

    biencoder = BiEncoder(model_name=model_path, is_training=False, long=long)
    # biencoder.load_state_dict(torch.load(model_path))

    return biencoder


def get_knn_candidate_map(
    evt_mention_map,
    split_mention_ids,
    biencoder,
    top_k,
    device="cuda",
    text_key="marked_sentence",
):
    """
    TODO: run bi-encoder prediction and get candidates cid_map = {eid: [c_ids]}
    TODO: make pairs of [(e_id, c) for eid, cs in cid_map.items() for c in cs]

    Parameters
    ----------
    evt_mention_map: Dict
    split_mention_ids: List[str]
    biencoder: BiEncoder
    top_k: int

    Returns
    -------
    List[(str, str)]
    """
    biencoder.eval()
    biencoder.to(device)

    tokenizer = biencoder.tokenizer
    m_start_id = biencoder.start_id
    m_end_id = biencoder.end_id

    # tokenize the event mentions in the split
    tokenized_dev_dict = tokenize_bi(
        tokenizer, split_mention_ids, evt_mention_map, m_end_id, text_key=text_key
    )
    split_dataset = Dataset.from_dict(tokenized_dev_dict).with_format("torch")
    split_dataset = split_dataset.map(
        lambda batch: get_arg_attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True,
        batch_size=1,
    )  # to include global attention mask and attention mask for the event mentions
    split_dataloader = DataLoader(split_dataset, batch_size=1, shuffle=False)

    faiss_db = VectorDatabase()
    dev_index, _ = create_faiss_db(split_dataset, biencoder, device=device)
    faiss_db.set_index(dev_index, split_dataset)

    result_list = []

    with torch.no_grad():
        for batch in tqdm(split_dataloader, desc="Biencoder prediction"):
            embeddings = process_batch(batch, biencoder, device)
            neighbor_index_list = faiss_db.get_nearest_neighbors(embeddings, top_k)[0][
                1:
            ]  # remove the first one which is the query itself
            candidate_ids = [i["mention_id"] for i in neighbor_index_list]
            result_list.append((batch["mention_id"][0], candidate_ids))

    return result_list


def biencoder_nn(
    dataset_folder, split, model_name, long, top_k=10, device="cuda", text_key=None
):
    """
    Generate mention pairs using the k-nearest neighbor approach of Held et al. 2021.

    config file: model name, path,

    Parameters
    ----------
    dataset_folder: str
    split: str
    model_name: str
    top_k: int
    long: bool

    Returns
    -------

    """
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    evt_mention_map = {
        m_id: m for m_id, m in mention_map.items() if m["men_type"] == "evt"
    }
    split_mention_ids = [
        key for key, val in evt_mention_map.items() if val["split"] == split
    ]

    biencoder = load_biencoder(model_name, long)

    # run the biencoder k-nn on the split
    all_mention_pairs = get_knn_candidate_map(
        evt_mention_map,
        split_mention_ids,
        biencoder,
        top_k,
        text_key=text_key,
        device=device,
    )

    return all_mention_pairs


if __name__ == "__main__":
    app()
