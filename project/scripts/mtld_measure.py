from collections import defaultdict
from lexical_diversity import lex_div as ld

import numpy as np
import pickle
import typer


app = typer.Typer()


def mtld_calculation(tokens, ttr_threshold=0.72):
    """
    Calculate the MTLD (Measure of Textual Lexical Diversity) for a given list of tokens.

    :param tokens: A list of tokens (words) from the text.
    :param ttr_threshold: The TTR threshold to consider for dividing the text. Default is 0.72.
    :return: The MTLD value as a float.
    """

    def mtld_segment(tokens, threshold):
        """
        Helper function to calculate MTLD for one direction.
        """
        token_count = 0
        type_count = 0
        factors = 0
        types = set()

        for token in tokens:
            token_count += 1
            if token not in types:
                type_count += 1
                types.add(token)
            if type_count / token_count < threshold:
                factors += 1
                token_count = 0
                type_count = 0
                types = set()

        # Avoid division by zero
        if token_count > 0:
            factors += 1

        return len(tokens) / factors

    forward_mtld = mtld_segment(tokens, ttr_threshold)
    reverse_mtld = mtld_segment(tokens[::-1], ttr_threshold)

    # Calculate the harmonic mean of the forward and reverse MTLD values
    mtld = 2 * (forward_mtld * reverse_mtld) / (forward_mtld + reverse_mtld)

    return mtld


def calculate_ds_measure(tokens):
    """
    Calculate D's measure of lexical diversity.
    :param tokens: A list of tokens (words) from the text.
    :return: D's measure as a float.
    """
    N = len(tokens)  # Total number of tokens
    V = len(set(tokens))  # Number of unique types

    # Calculate TTR (Type-Token Ratio)
    TTR = V / N

    # Calculate mean and standard deviation of token lengths
    token_lengths = [len(token) for token in tokens]
    mean_length = np.mean(token_lengths)
    std_dev_length = np.std(token_lengths)

    # Calculate D's measure (simplified version for demonstration)
    # Note: The actual calculation of D's measure might involve more complex statistical adjustments.
    D = (TTR - mean_length) / std_dev_length

    return D

@app.command()
def main(mention_map_path: str, split: str):
    mention_map = pickle.load(open(mention_map_path, "rb"))
    evt_map = {
        m_id: men
        for m_id, men in mention_map.items()
        if men["men_type"] == "evt" and men["split"] == split
    }

    clus2m_id = defaultdict(list)
    for m_id, mention in evt_map.items():
        clus_id = str(mention["gold_cluster"])
        if "ACT" in clus_id or "NEG" in clus_id:
            clus2m_id[clus_id].append(m_id)

    texts = [
        [
            t.strip()
            for m in mids
            for t  in evt_map[m]["mention_text"].strip().lower().split() if t.strip() != ""
        ]
        for mids in clus2m_id.values()
    ]

    overall_tokens = [w for t in texts for w in t if len(t) > 1]

    mtld_vals = np.array([ld.mtld(text) for text in texts if len(text) > 1])
    text_counts = np.array([len(text) for text in texts if len(text) > 1])

    total = np.sum(text_counts)


    print("MTLD weighted avg", np.sum(mtld_vals * text_counts / total))
    print("MTLD overall", ld.mtld(overall_tokens))


if __name__ == "__main__":
    app()
