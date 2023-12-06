import pickle
import torch
import typer

from pathlib import Path
from tqdm import tqdm

from .nn_method.models import CrossEncoder
from .nn_method.helper import forward_ab, tokenize_ce
from .prediction import evaluate, get_biencoder_knn
from .heuristic import get_lh_pairs


app = typer.Typer()


# ------------------ HELPERS ------------------ #


def predict_ce(parallel_model, dev_ab, dev_ba, device, batch_size):
    n = dev_ab["input_ids"].shape[0]
    indices = list(range(n))
    # new_batch_size = batching(n, batch_size, len(device_ids))
    # batch_size = new_batch_size
    all_scores_ab = []
    all_scores_ba = []
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Predicting"):
            batch_indices = indices[i : i + batch_size]
            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)
            all_scores_ab.append(scores_ab.detach().cpu())
            all_scores_ba.append(scores_ba.detach().cpu())

    return torch.cat(all_scores_ab), torch.cat(all_scores_ba)


def predict_trained_model(
    mention_map,
    model_name,
    linear_weights_path,
    test_pairs,
    text_key="bert_doc",
    max_sentence_len=1024,
    long=True,
    device="cuda:0",
    batch_size=256,
):
    device = torch.device(device)
    device_ids = list(range(1))
    linear_weights = torch.load(linear_weights_path)
    scorer_module = CrossEncoder(
        is_training=False,
        model_name=model_name,
        long=long,
        linear_weights=linear_weights,
    ).to(device)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    tokenizer = parallel_model.module.tokenizer

    test_ab, test_ba = tokenize_ce(
        tokenizer,
        test_pairs,
        mention_map,
        parallel_model.module.end_id,
        text_key=text_key,
        max_sentence_len=max_sentence_len,
    )

    scores_ab, scores_ba = predict_ce(
        parallel_model, test_ab, test_ba, device, batch_size=batch_size
    )

    return scores_ab, scores_ba


def get_ce_scores(
    mention_map,
    mention_pairs,
    ce_folder,
    text_key="marked_sentence",
    max_sentence_len=1024,
    is_long=True,
    device="cuda:0",
):
    linear_weights_path = ce_folder + "/linear.chkpt"
    bert_path = ce_folder + "/bert"

    scores_ab, scores_ba = predict_trained_model(
        mention_map,
        bert_path,
        linear_weights_path,
        mention_pairs,
        text_key,
        max_sentence_len,
        long=is_long,
        device=device,
    )
    return scores_ab, scores_ba


def run_ce(
    mention_map,
    split_mention_ids,
    mention_pairs,
    ce_folder,
    text_key="marked_sentence",
    max_sentence_len=1024,
    is_long=True,
    device="cuda:0",
    ce_score_file: Path = None,
    ce_threshold: float = 0.5,
    ce_force: bool = False,
):
    # generate the intermediate ce_scores to be used for clustering
    if ce_score_file.exists() and not ce_force:
        ce_scores_ab, ce_scores_ba = pickle.load(open(ce_score_file, "rb"))
    else:
        ce_scores_ab, ce_scores_ba = get_ce_scores(
            mention_map,
            mention_pairs,
            ce_folder,
            text_key=text_key,
            is_long=is_long,
            max_sentence_len=max_sentence_len,
            device=device,
        )
        pickle.dump((ce_scores_ab, ce_scores_ba), open(ce_score_file, "wb"))

    predictions = (ce_scores_ab + ce_scores_ba) / 2
    similarities = torch.squeeze(predictions) > ce_threshold

    scores = evaluate(
        mention_map,
        split_mention_ids,
        mention_pairs,
        similarities,
        tmp_folder=ce_folder,
    )
    print(scores)


# ------------------------- Commands ----------------------- #


@app.command()
def run_knn_lh_bert_pipeline(
    dataset_folder: str,
    split: str,
    model_name: str,
    ce_folder: str,
    knn_file: Path = None,
    top_k: int = 10,
    lh: str = "lh",
    lh_threshold: float = 0.15,
    device: str = "cuda:0",
    max_sentence_len: int = None,
    is_long: bool = False,
    ce_text_key: str = "marked_sentence",
    ce_score_file: Path = None,
    ce_threshold: float = 0.5,
    ce_force: bool = False,
):
    # read split mention map
    ensure_path(ce_score_file)
    ensure_path(knn_file)
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    split_mention_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == split
    ]

    # generate target-candidates map
    if knn_file and knn_file.exists():
        knn_map = pickle.load(open(knn_file, "rb"))
    else:
        knn_map = get_biencoder_knn(
            dataset_folder,
            split,
            model_name,
            knn_file,
            ce_text_key=ce_text_key,
            top_k=100,
            device=device,
            long=is_long,
        )

    # generate target-candidate knn mention pairs
    mention_pairs = set()
    for e_id, c_ids in knn_map:
        for c_id in c_ids[:top_k]:
            if e_id > c_id:
                e_id, c_id = c_id, e_id
            mention_pairs.add((e_id, c_id))

    print("After knn", len(mention_pairs))
    # generate LH mention pairs
    m_pairs, _ = get_lh_pairs(mention_map, split, lh, lh_threshold)
    for m1, m2 in m_pairs[0]:
        if m1 > m2:
            m1, m2 = m2, m1
        mention_pairs.add((m1, m2))

    print("after LH", len(mention_pairs))
    mention_pairs = sorted(list(mention_pairs))

    run_ce(
        mention_map,
        split_mention_ids,
        mention_pairs,
        ce_folder,
        ce_text_key,
        max_sentence_len,
        is_long,
        device,
        ce_score_file,
        ce_threshold,
        ce_force,
    )


@app.command()
def run_lh_bert_pipeline(
    dataset_folder: str,
    split: str,
    ce_folder: str,
    device: str = "cuda:0",
    max_sentence_len: int = None,
    is_long: bool = False,
    lh: str = "lh",
    lh_threshold: float = 0.05,
    ce_score_file: Path = None,
    ce_text_key: str = "marked_sentence",
    ce_threshold: float = 0.5,
    ce_force: bool = False,
):
    ensure_path(ce_score_file)
    # read split mention map
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    split_mention_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == split
    ]

    m_pairs, _ = get_lh_pairs(mention_map, split, lh, lh_threshold)

    # get the true positives and false positives
    mention_pairs = m_pairs[0] + m_pairs[1]

    run_ce(
        mention_map,
        split_mention_ids,
        mention_pairs,
        ce_folder,
        ce_text_key,
        max_sentence_len,
        is_long,
        device,
        ce_score_file,
        ce_threshold,
        ce_force,
    )


def ensure_path(file: Path):
    if not file.parent.exists():
        file.parent.mkdir(parents=True)


@app.command()
def run_knn_bert_pipeline(
    dataset_folder: str,
    split: str,
    model_name: str,
    ce_folder: str,
    knn_file: Path = None,
    top_k: int = 10,
    device: str = "cuda:0",
    max_sentence_len: int = None,
    is_long: bool = False,
    ce_text_key: str = "marked_sentence",
    ce_score_file: Path = None,
    ce_threshold: float = 0.5,
    ce_force: bool = False,
):
    # read split mention map
    ensure_path(ce_score_file)
    ensure_path(knn_file)
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    split_mention_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == split
    ]

    # generate target-candidates map
    if knn_file and knn_file.exists():
        knn_map = pickle.load(open(knn_file, "rb"))
    else:
        knn_map = get_biencoder_knn(
            dataset_folder,
            split,
            model_name,
            knn_file,
            ce_text_key=ce_text_key,
            top_k=100,
            device=device,
            long=is_long,
        )

    # generate target-candidate mention pairs
    mention_pairs = set()
    for e_id, c_ids in knn_map:
        for c_id in c_ids[:top_k]:
            if e_id > c_id:
                e_id, c_id = c_id, e_id
            mention_pairs.add((e_id, c_id))
    mention_pairs = sorted(list(mention_pairs))

    run_ce(
        mention_map,
        split_mention_ids,
        mention_pairs,
        ce_folder,
        ce_text_key,
        max_sentence_len,
        is_long,
        device,
        ce_score_file,
        ce_threshold,
        ce_force,
    )


if __name__ == "__main__":
    app()
