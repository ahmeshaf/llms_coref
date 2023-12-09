import pickle
import torch

from pathlib import Path

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Typer

from .bert.helper import (
    get_arg_attention_mask_wrapper,
    get_arg_attention_mask_wrapper_ce,
    process_batch,
    create_faiss_db,
    VectorDatabase,
    tokenize_bi,
    tokenize_ce
)
from .bert.bi_encoder import BiEncoder
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
    dev_index, _ = create_faiss_db(split_dataset, crossencoder, device=device) # generate embeddings 
    faiss_db.set_index(dev_index, split_dataset)
    
    result_list = []
    
    with torch.no_grad():
        for batch in tqdm(split_dataloader): 
            embeddings = process_batch(batch, crossencoder, device)
            # remove the first one which is the query itself
            neighbor_list = faiss_db.get_nearest_neighbors(embeddings, top_k)[0][
                1:
            ]
            candidate_ids = [i["mention_pairs"] for i in neighbor_list] # will be tuple of m1, m2
            result_list.append((batch["mention_pairs"], candidate_ids))

    return result_list



def get_knn_candidate_map(
    evt_mention_map,
    split_mention_ids,
    biencoder,
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
    biencoder.eval()
    biencoder.to(device)

    tokenizer = biencoder.tokenizer
    m_start_id = biencoder.start_id
    m_end_id = biencoder.end_id

    # tokenize the event mentions in the split
    tokenized_dict = tokenize_bi(
        tokenizer, split_mention_ids, evt_mention_map, m_end_id, text_key=text_key
    )
    split_dataset = Dataset.from_dict(tokenized_dict).with_format("torch")
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
    device: str = "cuda",
    long: bool = False,
):
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    candidate_map = biencoder_nn(
        dataset_folder, split, model_path, long, 100, device, text_key=ce_text_key
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
            device,
            long,
        )
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    knn_map = pickle.load(open(output_file, "rb"))
    # generate target-candidate knn mention pairs
    mention_pairs = set()
    for e_id, c_ids in knn_map:
        # filter out c_ids not in the same topic and in the same sentence.
        c_ids_clean = [
            c_id
            for c_id in c_ids
            if mention_map[e_id]["topic"] == mention_map[c_id]["topic"]
            and (mention_map[e_id]["doc_id"], mention_map[e_id]["sentence_id"])
            != (mention_map[c_id]["doc_id"], mention_map[c_id]["sentence_id"])
        ]

        for c_id in c_ids_clean[:top_k]:
            p = tuple(sorted((e_id, c_id)))
            mention_pairs.add(p)
    return mention_pairs

# TODO: remove since no code is using this.
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
    print(top_k)
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
