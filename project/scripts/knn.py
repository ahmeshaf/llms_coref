import os.path
import pickle
from collections import defaultdict
from typing import List, Union, Tuple

import numpy as np
import torch

from pathlib import Path

from datasets import Dataset
from evaluate import load
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Typer

from .bert.models import CrossEncoder
from .bert.helper import (
    get_arg_attention_mask_wrapper,
    get_arg_attention_mask_wrapper_ce,
    process_batch,
    create_faiss_db,
    VectorDatabase,
    tokenize_bi,
    tokenize_ce,
)
from .bert.bi_encoder import BiEncoder
from .helper import ensure_path, get_split_ids

app = Typer()


def load_model_from_path(model_class, biencoder_path, training=False, long=False):
    bert_path = biencoder_path + "/bert"
    linear_weights = None
    linear_path = biencoder_path + "/linear.pt"
    if os.path.exists(linear_path):
        linear_weights = torch.load(linear_path)
    return model_class(
        model_name=bert_path,
        linear_weights=linear_weights,
        is_training=training,
        long=long,
    )


def load_biencoder_model_linear(bert_path, linear_weights, training=False, long=False):
    return BiEncoder(
        model_name=bert_path,
        linear_weights=linear_weights,
        is_training=training,
        long=long,
    )


def get_knn_candidate_map_ce(
    evt_mention_map,
    mention_pairs,
    crossencoder,
    top_k,
    device="cuda",
    text_key="marked_sentence",
):
    crossencoder.eval()
    crossencoder.to(device)

    tokenizer = crossencoder.tokenizer
    m_start_id = crossencoder.start_id
    m_end_id = crossencoder.end_id
    # tokenize the event pairs in the split
    # only takes the first element of the tokenize_ce result, order: a b
    tokenized_dict, _ = tokenize_ce(
        tokenizer, mention_pairs, evt_mention_map, m_end_id, text_key=text_key
    )
    split_dataset = Dataset.from_dict(tokenized_dict).with_format("torch")
    split_dataset = split_dataset.map(
        lambda batch: get_arg_attention_mask_wrapper_ce(batch, m_start_id, m_end_id),
        batched=True,
        batch_size=128,
    )  # to include global attention mask and attention mask for the event pairs

    split_dataloader = DataLoader(split_dataset, batch_size=128, shuffle=False)

    faiss_db = VectorDatabase()
    dev_index, _ = create_faiss_db(
        split_dataset, crossencoder, device=device
    )  # generate embeddings
    faiss_db.set_index(dev_index, split_dataset)

    result_list = []

    with torch.no_grad():
        for batch in tqdm(split_dataloader):
            embeddings = process_batch(batch, crossencoder, device)
            # remove the first one which is the query itself
            neighbor_list = faiss_db.get_nearest_neighbors(embeddings, top_k)
            candidate_ids = [
                [tuple(cand["unit_ids"]) for cand in cands[1:]]
                for cands in neighbor_list
            ]  # will be tuple of m1, m2
            for j in range(len(batch["unit_ids"][0])):
                target = (batch["unit_ids"][0][j], batch["unit_ids"][1][j])
                candidates = candidate_ids[j]
                result_list.append((target, candidates))

            # target_id = (target_id[0][0], target_id[1][0])
            # result_list.append((target_id, candidate_ids))

    return result_list


@torch.no_grad
def get_knn(
    mention_map,
    train_ids,
    test_ids,
    model,
    device="cuda:0",
    k=10,
    batch_size=64,
    text_key="marked_sentence",
    max_sentence_len=256,
):
    device = torch.device(device)
    model.to(device)

    if isinstance(model, BiEncoder):
        tokenize_method = tokenize_bi
        attention_mask_wrapper = get_arg_attention_mask_wrapper
    elif isinstance(model, CrossEncoder):
        tokenize_method = tokenize_ce
        attention_mask_wrapper = get_arg_attention_mask_wrapper_ce
    else:
        raise Exception("Invalid Model Type. Use BiEncoder or CrossEncoder")

    tokenizer = model.tokenizer
    m_start_id = model.start_id
    m_end_id = model.end_id

    # tokenize the event mentions in the split
    tokenized_train = tokenize_method(
        tokenizer,
        train_ids,
        mention_map,
        m_end_id,
        text_key=text_key,
        max_sentence_len=max_sentence_len,
    )

    tokenized_test = tokenize_method(
        tokenizer,
        test_ids,
        mention_map,
        m_end_id,
        text_key=text_key,
        max_sentence_len=max_sentence_len,
    )

    if isinstance(tokenized_train, tuple):
        tokenized_train = tokenized_train[0]
    if isinstance(tokenized_test, tuple):
        tokenized_test = tokenized_test[0]

    train_dataset = Dataset.from_dict(tokenized_train).with_format("torch")
    # to include global attention mask and attention mask for the event mentions
    train_dataset = train_dataset.map(
        lambda batch: attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True,
        batch_size=batch_size,
    )

    faiss_db = VectorDatabase()
    train_index, _ = create_faiss_db(train_dataset, model, device=device)
    faiss_db.set_index(train_index, train_dataset)

    test_dataset = Dataset.from_dict(tokenized_test).with_format("torch")
    # to include global attention mask and attention mask for the event mentions
    test_dataset = test_dataset.map(
        lambda batch: attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True,
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    result_list = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="KNN Prediction"):
            embeddings = process_batch(batch, model, device)
            # remove the first one which is the query itself
            neighbor_index_list = faiss_db.get_nearest_neighbors(embeddings, k)
            for neighbors in neighbor_index_list:
                candidate_ids = [i["unit_ids"] for i in neighbors]
            result_list.append(candidate_ids)

    return result_list


@torch.no_grad
def get_knn_candidate_map(
    evt_mention_map,
    units,
    model,
    top_k,
    device="cuda",
    text_key="marked_sentence",
):
    """
    Parameters
    ----------

    Returns
    -------
    List[(str, str)]
    """
    model.eval()
    model.to(device)

    if isinstance(model, BiEncoder):
        tokenize_method = tokenize_bi
        attention_mask_wrapper = get_arg_attention_mask_wrapper
    elif isinstance(model, CrossEncoder):
        tokenize_method = tokenize_ce
        attention_mask_wrapper = get_arg_attention_mask_wrapper_ce
    else:
        raise Exception("Invalid Model Type. Use BiEncoder or CrossEncoder")

    tokenizer = model.tokenizer
    m_start_id = model.start_id
    m_end_id = model.end_id

    # tokenize the event mentions in the split
    tokenized_dict = tokenize_method(
        tokenizer, units, evt_mention_map, m_end_id, text_key=text_key
    )
    if isinstance(tokenized_dict, tuple):
        tokenized_dict = tokenized_dict[0]
    units_dataset = Dataset.from_dict(tokenized_dict).with_format("torch")
    units_dataset = units_dataset.map(
        lambda batch: attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True,
        batch_size=1,
    )  # to include global attention mask and attention mask for the event mentions
    units_dataloader = DataLoader(units_dataset, batch_size=1, shuffle=False)

    faiss_db = VectorDatabase()
    dev_index, _ = create_faiss_db(units_dataset, model, device=device)
    faiss_db.set_index(dev_index, units_dataset)

    result_list = []

    with torch.no_grad():
        for batch in tqdm(units_dataloader, desc="KNN Prediction"):
            embeddings = process_batch(batch, model, device)
            # remove the first one which is the query itself
            neighbor_index_list = faiss_db.get_nearest_neighbors(embeddings, top_k)[0][
                1:
            ]
            candidate_ids = [i["unit_ids"] for i in neighbor_index_list]
            result_list.append((batch["unit_ids"][0], candidate_ids))

    return result_list


def encoder_nn(mention_map, units, model, top_k=10, device="cuda", text_key=None):
    """
    Generate mention pairs using the k-nearest neighbor approach of Held et al. 2021.

    Returns
    -------

    """

    candidate_map = get_knn_candidate_map
    if isinstance(model, CrossEncoder):
        candidate_map = get_knn_candidate_map_ce

    # run the model k-nn on the split
    all_mention_pairs = candidate_map(
        mention_map,
        units,
        model,
        top_k,
        text_key=text_key,
        device=device,
    )

    return all_mention_pairs


def get_knn_map(
    mention_map: dict,
    units: Union[List[str], List[Tuple[str, str]]],
    model: Union[BiEncoder, CrossEncoder],
    output_file: Path,
    ce_text_key: str = "marked_sentence",
    device: str = "cuda",
):
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    candidate_map = encoder_nn(
        mention_map, units, model, 100, device, text_key=ce_text_key
    )
    print(len(candidate_map))
    pickle.dump(candidate_map, open(output_file, "wb"))
    return candidate_map


def get_knn_pairs(
    mention_map: dict,
    units: Union[List[str], List[Tuple[str, str]]],
    model: Union[CrossEncoder, BiEncoder],
    output_file: Path,
    ce_text_key: str = "marked_sentence",
    top_k: int = 10,
    device: str = "cuda",
    force: bool = False,
):
    if not output_file.exists() or force:
        get_knn_map(
            mention_map,
            units,
            model,
            output_file,
            ce_text_key,
            device,
        )
    knn_map = pickle.load(open(output_file, "rb"))
    # generate target-candidate knn mention pairs
    target_candidate_pairs = set()
    knn_map = [
        (
            e_id,
            [
                cid
                for cid in c_ids
                if mention_map[e_id]["topic"] == mention_map[cid]["topic"]
                and (mention_map[e_id]["doc_id"], mention_map[e_id]["sentence_id"])
                != (mention_map[cid]["doc_id"], mention_map[cid]["sentence_id"])
            ],
        )
        for e_id, c_ids in knn_map
    ]
    for e_id, c_ids in knn_map:
        for c_id in c_ids[:top_k]:
            p = tuple(sorted((e_id, c_id)))
            target_candidate_pairs.add(p)
    return list(target_candidate_pairs)


@app.command()
def save_knn_few_shot_pairs(
    mention_map_file: Path,
    train_pairs_path: Path,
    split: str,
    save_file_path: Path,
    bi_encoder_path: str,
    text_key: str = "marked_sentence",
    batch_size: int = 64,
    max_sentence_len: int = 256,
):
    ensure_path(save_file_path)
    mention_map = pickle.load(open(mention_map_file, "rb"))
    test_ids = get_split_ids(mention_map, split)

    train_ids = get_split_ids(mention_map, "train")

    train_mention_pairs = pickle.load(open(train_pairs_path, "rb"))

    biencoder_model = load_biencoder_model_linear(
        bert_path=bi_encoder_path + "/bert",
        linear_weights=torch.load(bi_encoder_path + "/linear.pt"),
        long=False,
        training=False,
    )

    train2pos_pairs = defaultdict(set)
    train2neg_pairs = defaultdict(set)
    for m1, m2 in train_mention_pairs:
        if mention_map[m1]["gold_cluster"] == mention_map[m2]["gold_cluster"]:
            train2pos_pairs[m1].add((m1, m2))
            train2pos_pairs[m2].add((m1, m2))
        else:
            train2neg_pairs[m1].add((m1, m2))
            train2neg_pairs[m2].add((m1, m2))

    knn_map = get_knn(
        mention_map,
        train_ids,
        test_ids,
        biencoder_model,
        device="cuda:0",
        k=10,
        batch_size=batch_size,
        text_key=text_key,
        max_sentence_len=max_sentence_len,
    )
    pass

    m_id2k_pairs = {t_id: candidates for t_id, candidates in zip(test_ids, knn_map)}

    pickle.dump(m_id2k_pairs, open(save_file_path, "wb"))


@torch.no_grad
def get_mention_pair_similarity_bertscore(
    mention_pairs_inds,
    retrieve_mention_ids,
    store_mention_ids,
    mention_map,
    text_key,
    batch_size=256,
    big_batch_size=100000,
):
    all_sent_scores = []
    all_mention_scores = []

    def get_b_sent(mention_map, m_id):
        return (
            mention_map[m_id]["mention_text"] + " [SEP] " + mention_map[m_id][text_key]
        )

    for i in tqdm(
        range(0, len(mention_pairs_inds), big_batch_size),
        total=len(mention_pairs_inds) / big_batch_size,
    ):
        mention_pairs = [
            (retrieve_mention_ids[j], store_mention_ids[k])
            for j, k in mention_pairs_inds[i : i + big_batch_size]
        ]
        m1_sentences = [get_b_sent(mention_map, m1).lower() for m1, m2 in mention_pairs]
        m2_sentences = [get_b_sent(mention_map, m2).lower() for m1, m2 in mention_pairs]

        m1_texts = [mention_map[m1]["mention_text"].lower() for m1, m2 in mention_pairs]
        m2_texts = [mention_map[m2]["mention_text"].lower() for m1, m2 in mention_pairs]

        bertscore = load("bertscore")
        pairwise_scores_sent = bertscore.compute(
            predictions=m1_sentences,
            references=m2_sentences,
            lang="en",
            model_type="distilbert-base-uncased",
            num_layers=6,
            verbose=True,
            device="cuda:0",
            batch_size=batch_size,
        )

        pairwise_scores_texts = bertscore.compute(
            predictions=m1_texts,
            references=m2_texts,
            lang="en",
            model_type="distilbert-base-uncased",
            num_layers=6,
            verbose=True,
            device="cuda:0",
            batch_size=batch_size,
        )

        sent_scores = np.array(pairwise_scores_sent["f1"])
        m_text_scores = np.array(pairwise_scores_texts["f1"])
        all_mention_scores.extend(m_text_scores.tolist())
        all_sent_scores.extend(sent_scores.tolist())

    return all_mention_scores, all_sent_scores


@app.command()
def save_knn_pairs_bert_score(
    mention_map_file: Path,
    store_split: str,
    retrieve_split: str,
    pairs_output_file: Path,
    text_key: str = "marked_sentence",
    batch_size: int = 64,
    top_k: int = 5,
    big_batch_size: int = 100000,
    same_topic: bool = False,
):
    ensure_path(pairs_output_file)
    mention_map = pickle.load(open(mention_map_file, "rb"))

    store_mention_ids = get_split_ids(mention_map, store_split)
    retrieve_mention_ids = get_split_ids(mention_map, retrieve_split)

    mention_pairs_inds = [
        (i, j)
        for i in range(len(retrieve_mention_ids))
        for j in range(len(store_mention_ids))
    ]
    if same_topic:
        mention_pairs_inds = [
            (i, j)
            for i, j in mention_pairs_inds
            if mention_map[retrieve_mention_ids[i]]["topic"]
            == mention_map[store_mention_ids[j]]["topic"]
        ]
    print(len(mention_pairs_inds))
    lemma_scores, sentence_scores = get_mention_pair_similarity_bertscore(
        mention_pairs_inds,
        retrieve_mention_ids,
        store_mention_ids,
        mention_map,
        text_key,
        batch_size,
        big_batch_size=big_batch_size,
    )
    score_map = defaultdict(list)
    for (i, j), l_score, s_score in zip(
        mention_pairs_inds, lemma_scores, sentence_scores
    ):
        score_map[i].append((j, 0.7 * l_score + 0.3 * s_score))

    score_map = {
        i: sorted(vals, key=lambda x: x[1], reverse=True)[:top_k]
        for i, vals in score_map.items()
    }

    output_pairs = set()
    for m1, vals in score_map.items():
        for val in vals:
            pair = tuple(sorted([retrieve_mention_ids[m1], store_mention_ids[val[0]]]))
            output_pairs.add(pair)

    print("Saved Pairs", len(output_pairs))
    output_pairs = list(output_pairs)
    pickle.dump(output_pairs, open(pairs_output_file, "wb"))


@app.command()
def save_knn_mention_pairs(
    mention_map_file: str,
    split: str,
    model_path: str,
    knn_output_file: Path,
    pairs_output_file: Path,
    ce_text_key: str = "marked_sentence",
    top_k: int = 10,
    device: str = "cuda",
    long: bool = False,
    force: bool = False,
):
    ensure_path(knn_output_file)
    ensure_path(pairs_output_file)

    mention_map = pickle.load(open(mention_map_file, "rb"))
    split_mention_ids = [
        m_id
        for m_id, men in mention_map.items()
        if men["men_type"] == "evt" and men["split"] == split
    ]
    model = load_model_from_path(BiEncoder, model_path, training=False, long=long)
    mention_pairs = get_knn_pairs(
        mention_map,
        split_mention_ids,
        model,
        knn_output_file,
        ce_text_key,
        top_k,
        device,
        force,
    )
    print("mention pairs", len(mention_pairs))
    pickle.dump(mention_pairs, open(pairs_output_file, "wb"))


@app.command()
def save_knn_mention_pairs_ce(
    mention_map_file: str,
    mention_pairs_path: Path,
    model_path: str,
    knn_output_file: Path,
    pairs_output_file: Path,
    ce_text_key: str = "marked_sentence",
    top_k: int = 10,
    device: str = "cuda",
    force: bool = False,
):
    ensure_path(knn_output_file)
    ensure_path(pairs_output_file)
    mention_map = pickle.load(open(mention_map_file, "rb"))
    mention_pairs = pickle.load(open(mention_pairs_path, "rb"))
    # mention_pairs = mention_pairs[:200]
    model = load_model_from_path(CrossEncoder, model_path, training=False, long=False)
    mention_pairs_res = get_knn_pairs(
        mention_map,
        mention_pairs,
        model,
        knn_output_file,
        ce_text_key,
        top_k,
        device,
        force,
    )
    print("mention pairs", len(mention_pairs_res))
    pickle.dump(mention_pairs_res, open(pairs_output_file, "wb"))


@app.command()
def merge_mention_pairs(p1_path: Path, p2_path: Path, outfile_path: Path):
    ensure_path(outfile_path)
    pairs1 = list(pickle.load(open(p1_path, "rb")))
    pairs2 = list(pickle.load(open(p2_path, "rb")))
    print("p1", len(pairs1))
    print("p2", len(pairs2))

    merged = set()
    for m1, m2 in pairs1 + pairs2:
        if m1 > m2:
            merged.add((m1, m2))
        else:
            merged.add((m2, m1))
    print("merged", len(merged))
    pickle.dump(merged, open(outfile_path, "wb"))


if __name__ == "__main__":
    app()
