import pickle
import torch
import typer

from pathlib import Path
from tqdm import tqdm

from .bert.models import CrossEncoder
from .bert.helper import forward_ab, tokenize_ce
from .helper import evaluate, ensure_path

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
    batch_size=128,
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
    is_long=False,
    device="cuda:0",
    ce_score_file: Path = None,
    ce_threshold: float = 0.5,
    ce_force: bool = False,
):
    # generate the intermediate ce_scores to be used for clustering
    if ce_score_file.exists() and not ce_force:
        mention_pairs, ce_scores_ab, ce_scores_ba = pickle.load(
            open(ce_score_file, "rb")
        )
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
        mention_pairs = [tuple(sorted(p)) for p in mention_pairs]
        pickle.dump(
            (mention_pairs, ce_scores_ab, ce_scores_ba), open(ce_score_file, "wb")
        )

    predictions = torch.squeeze( (ce_scores_ab + ce_scores_ba) / 2 )
    # similarities = torch.squeeze(predictions) > ce_threshold

    scores = evaluate(mention_map, split_mention_ids, mention_pairs, predictions, tmp_folder=ce_folder)
    print(scores)


# ------------------------- Commands ----------------------- #


@app.command()
def run_ce_mention_pairs(
    dataset_folder: str,
    split: str,
    mention_pairs_path: Path,
    ce_folder,
    text_key="marked_sentence",
    max_sentence_len: int = 1024,
    is_long: bool = False,
    device="cuda:0",
    ce_score_file: Path = None,
    ce_threshold: float = 0.5,
    ce_force: bool = False,
):
    ensure_path(ce_score_file)
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))

    split_mention_ids = {
        m_id
        for m_id, men in mention_map.items()
        if men["split"] == split and men["men_type"] == "evt"
    }

    mention_pairs = sorted(pickle.load(open(mention_pairs_path, "rb")))

    mention_pairs = [
        (m1, m2)
        for m1, m2 in mention_pairs
        if mention_map[m1]["topic"] == mention_map[m2]["topic"]
           and (mention_map[m1]["doc_id"], mention_map[m1]["sentence_id"])
           != (mention_map[m2]["doc_id"], mention_map[m2]["sentence_id"])
    ]

    run_ce(
        mention_map,
        split_mention_ids,
        mention_pairs,
        ce_folder,
        text_key,
        max_sentence_len,
        is_long,
        device,
        ce_score_file,
        ce_threshold,
        ce_force,
    )


if __name__ == "__main__":
    app()
