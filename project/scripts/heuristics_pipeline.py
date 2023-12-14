import pickle

import numpy as np
from typer import Typer

from .helper import evaluate

app = Typer()


def get_scores(mention_map, split, mention_pairs, oracle=False):
    evt_mention_map = {
        m_id: men
        for m_id, men in mention_map.items()
        if men["men_type"] == "evt" and men["split"] == split
    }
    split_mention_ids = list(evt_mention_map.keys())
    mention_pairs = [
        (m1, m2)
        for m1, m2 in mention_pairs
        if mention_map[m1]["topic"] == mention_map[m2]["topic"]
        and (mention_map[m1]["doc_id"], mention_map[m1]["sentence_id"])
        != (mention_map[m2]["doc_id"], mention_map[m2]["sentence_id"])
    ]

    similarities = np.ones(len(mention_pairs))
    if oracle is True:
        similarities = np.array(
            [
                mention_map[m1]["gold_cluster"] == mention_map[m2]["gold_cluster"]
                for m1, m2 in mention_pairs
            ]
        )
    # breakpoint()
    scores = evaluate(
        mention_map, split_mention_ids, mention_pairs, similarities, tmp_folder="/tmp/"
    )
    return scores


@app.command()
def run_mp_pipeline(mention_map_file, split, mention_pairs_file, oracle: bool = False):
    mention_map = pickle.load(open(mention_map_file, "rb"))
    mention_pairs = pickle.load(open(mention_pairs_file, "rb"))
    print(get_scores(mention_map, split, mention_pairs, oracle))


if __name__ == "__main__":
    app()