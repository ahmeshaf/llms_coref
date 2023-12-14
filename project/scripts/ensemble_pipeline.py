import pickle

import numpy as np
import torch
from typer import Typer

from .helper import evaluate

app = Typer()


def get_a_b_scores(mention_map_file, split, score_file_a, score_file_b, threshold):
    mention_map = pickle.load(open(mention_map_file, "rb"))
    split_ids = [
        m_id
        for m_id, men in mention_map.items()
        if men["men_type"] == "evt" and men["split"] == split
    ]

    mention_pairs_a, scores_a_ab, scores_a_ba = pickle.load(open(score_file_a, "rb"))
    mention_pairs_b, scores_b_ab, scores_b_ba = pickle.load(open(score_file_b, "rb"))

    p_a_map = {
        p: (score_ab, score_ba)
        for p, score_ab, score_ba in zip(mention_pairs_a, scores_a_ab, scores_a_ba)
    }
    p_b_map = {
        p: (score_ab, score_ba)
        for p, score_ab, score_ba in zip(mention_pairs_b, scores_b_ab, scores_b_ba)
    }

    print("p1", len(mention_pairs_a))
    print("pw", len(mention_pairs_b))

    all_pairs = sorted(set(mention_pairs_a + mention_pairs_b))

    print("merged", len(all_pairs))

    a_scores = []
    b_scores = []

    for p in all_pairs:
        if p in p_a_map:
            a_scores.append(np.mean(p_a_map[p]))
        else:
            a_scores.append(0)

        if p in p_b_map:
            b_scores.append(np.mean(p_b_map[p]))
        else:
            b_scores.append(0)

    a_scores = np.array(a_scores)
    b_scores = np.array(b_scores)

    similarities_a = a_scores > threshold
    similarities_b = b_scores > threshold

    return mention_map, split_ids, all_pairs, similarities_a, similarities_b


@app.command()
def and_clustering(
    mention_map_file,
    split,
    score_file_a,
    score_file_b,
    working_folder="/tmp/",
    threshold: float = 0.5,
):
    mention_map, split_ids, all_pairs, similarities_a, similarities_b = get_a_b_scores(
        mention_map_file, split, score_file_a, score_file_b, threshold
    )
    similarities_a_and_b = np.logical_and(similarities_a, similarities_b)
    scores = evaluate(mention_map, split_ids, all_pairs, similarities_a_and_b, tmp_folder=working_folder)
    print(scores)


@app.command()
def or_clustering(
    mention_map_file,
    split,
    score_file_a,
    score_file_b,
    working_folder="/tmp/",
    threshold: float = 0.5,
):
    mention_map, split_ids, all_pairs, similarities_a, similarities_b = get_a_b_scores(
        mention_map_file, split, score_file_a, score_file_b, threshold
    )
    similarities_a_and_b = np.logical_or(similarities_a, similarities_b)
    scores = evaluate(mention_map, split_ids, all_pairs, similarities_a_and_b, tmp_folder=working_folder)
    print(scores)


if __name__ == "__main__":
    app()
