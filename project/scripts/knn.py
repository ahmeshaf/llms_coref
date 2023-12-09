import pickle
from typing import List, Union, Tuple

import torch

from pathlib import Path

from datasets import Dataset
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
from .helper import ensure_path

app = Typer()


def load_model_from_path(model_class, biencoder_path, training=False, long=False):
    bert_path = biencoder_path + "/bert"
    linear_path = biencoder_path + "/linear.pt"
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
        batch_size=1,
    )  # to include global attention mask and attention mask for the event pairs

    split_dataloader = DataLoader(split_dataset, batch_size=1, shuffle=False)

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
            neighbor_list = faiss_db.get_nearest_neighbors(embeddings, top_k)[0][1:]
            candidate_ids = [
                tuple(i["unit_ids"]) for i in neighbor_list
            ]  # will be tuple of m1, m2
            target_id = batch["unit_ids"]
            target_id = (target_id[0][0], target_id[1][0])
            result_list.append((target_id, candidate_ids))

    return result_list


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


def encoder_nn(
    mention_map, units, model, top_k=10, device="cuda", text_key=None
):
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
    for e_id, c_ids in knn_map:
        # filter out c_ids not in the same topic and in the same sentence.
        # c_ids_clean = [
        #     c_id
        #     for c_id in c_ids
        #     if mention_map[e_id]["topic"] == mention_map[c_id]["topic"]
        #     and (mention_map[e_id]["doc_id"], mention_map[e_id]["sentence_id"])
        #     != (mention_map[c_id]["doc_id"], mention_map[c_id]["sentence_id"])
        # ]
        for c_id in c_ids[:top_k]:
            p = tuple(sorted((e_id, c_id)))
            target_candidate_pairs.add(p)
    return list(target_candidate_pairs)


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
    model = load_model_from_path(BiEncoder, model_path, training=False, long=False)
    mention_pairs = get_knn_pairs(
        mention_map,
        split_mention_ids,
        model,
        knn_output_file,
        ce_text_key,
        top_k,
        device,
        long,
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
    # mention_pairs = mention_pairs[:100]
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
