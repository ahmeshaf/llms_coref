import pickle
import torch

from pathlib import Path

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Typer

from .nn_method.helper import (
    get_arg_attention_mask_wrapper,
    process_batch,
    create_faiss_db,
    VectorDatabase,
    tokenize_bi,
)
from .nn_method.bi_encoder import BiEncoder
from .helper import ensure_path

app = Typer()


def load_biencoder_from_path(biencoder_path, training=False, long=False):
    bert_path = biencoder_path + "/bert"
    linear_path = biencoder_path + "/linear.pt"
    linear_weights = torch.load(linear_path)
    return BiEncoder(
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
            # remove the first one which is the query itself
            neighbor_index_list = faiss_db.get_nearest_neighbors(embeddings, top_k)[0][
                1:
            ]
            candidate_ids = [i["mention_id"] for i in neighbor_index_list]
            result_list.append((batch["mention_id"][0], candidate_ids))

    return result_list


def biencoder_nn(
    dataset_folder, split, model_name, long, top_k=10, device="cuda", text_key=None
):
    """
    Generate mention pairs using the k-nearest neighbor approach of Held et al. 2021.

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

    biencoder = load_biencoder_from_path(model_name, long)

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


def get_knn_map(
    dataset_folder: str,
    split: str,
    model_path: str,
    output_file: Path,
    ce_text_key: str = "marked_sentence",
    top_k: int = 10,
    device: str = "cuda",
    long: bool = False,
):
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    candidate_map = biencoder_nn(
        dataset_folder, split, model_path, long, top_k, device, text_key=ce_text_key
    )
    print(len(candidate_map))
    pickle.dump(candidate_map, open(output_file, "wb"))
    return candidate_map


def get_knn_pairs(
    dataset_folder: str,
    split: str,
    model_path: str,
    output_file: Path,
    ce_text_key: str = "marked_sentence",
    top_k: int = 10,
    device: str = "cuda",
    long: bool = False,
    force: bool = False,
):
    if not output_file.exists() or force:
        get_knn_map(
            dataset_folder,
            split,
            model_path,
            output_file,
            ce_text_key,
            top_k,
            device,
            long,
        )

    knn_map = pickle.load(open(output_file, "rb"))
    # generate target-candidate knn mention pairs
    mention_pairs = set()
    for e_id, c_ids in knn_map:
        for c_id in c_ids[:top_k]:
            if e_id > c_id:
                e_id, c_id = c_id, e_id
            mention_pairs.add((e_id, c_id))
    return mention_pairs


@app.command()
def save_knn_mention_pairs(
    dataset_folder: str,
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
    mention_pairs = get_knn_pairs(
        dataset_folder,
        split,
        model_path,
        knn_output_file,
        ce_text_key,
        top_k,
        device,
        long,
        force,
    )
    print("mention pairs", len(mention_pairs))
    pickle.dump(mention_pairs, open(pairs_output_file, "wb"))


if __name__ == "__main__":
    app()