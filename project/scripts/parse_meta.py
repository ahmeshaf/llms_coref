import copy
import difflib
import pickle
import os
import re
import spacy
import sys
import typer

from collections import defaultdict
from tqdm import tqdm

app = typer.Typer()


def load_pickle(file_path):
    """Load a pickle file from the given path."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, file_path):
    """Save data to a pickle file at the given path."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def process_meta_single(meta_single, doc_sent_map):
    """Process the meta_single dictionary and update doc_sent_map accordingly."""
    meta_sentence_key = "Metaphoric Sentence"
    meta_sentence_key_plural = "Metaphoric Sentences"
    meta_sentence_key_1 = "Metaphoric Sentence 1"
    the_meta_sk = "The Metaphoric Sentence"
    the_meta_sk_plu = "The Metaphoric Sentences"

    for (doc_id, sent_id), meta_val in meta_single.items():
        my_key = meta_sentence_key
        my_key_plu = meta_sentence_key_plural

        if the_meta_sk in meta_val:
            my_key = the_meta_sk
        if the_meta_sk_plu in meta_val:
            my_key_plu = the_meta_sk_plu

        if my_key_plu in meta_val:
            meta_val[meta_sentence_key] = meta_val[my_key_plu][0]
        elif my_key not in meta_val and meta_sentence_key_1 in meta_val:
            meta_val[meta_sentence_key] = meta_val[meta_sentence_key_1]

        if my_key in meta_val and isinstance(meta_val[my_key], list):
            meta_val[meta_sentence_key] = meta_val[my_key][0]

        if my_key in meta_val:
            meta_val[meta_sentence_key] = meta_val[my_key]

        doc_sent_map[doc_id][sent_id]["sentence"] = meta_val["Metaphoric Sentence"]


def find_most_similar(phrase, string_list):
    if phrase in string_list:
        return phrase
    # Initialize the highest similarity score and the most similar string
    highest_similarity = 0.0
    most_similar_string = None

    # Iterate through each string in the list
    for string in string_list:
        # Calculate the similarity score between the phrase and the current string
        similarity = difflib.SequenceMatcher(None, phrase, string).ratio()

        # If the current similarity score is higher than the highest recorded, update the highest score and the most similar string
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_string = string

    return most_similar_string


def find_most_similar_complete_word(phrase, sentence):
    if re.search(r"\b" + re.escape(phrase) + r"\b", sentence) is not None:
        return phrase, 1
    words = sentence.split()  # Split the sentence into words
    highest_similarity = 0.0
    most_similar_word = None

    # Function to generate n-grams from the list of words
    def generate_ngrams(words, n):
        ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        return ngrams

    # Check all n-grams for various lengths to ensure we consider combinations of words
    for n in range(
        1, min(len(words) + 1, 6)
    ):  # Assuming a max length of n-grams to 4 for performance
        for ngram in generate_ngrams(words, n):
            similarity = difflib.SequenceMatcher(None, phrase, ngram).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_word = ngram

    return most_similar_word, highest_similarity


def replace_first_occurrence(text, word, replacement):
    """
    Replaces the first occurrence of a word in a given text with a specified replacement word,
    and indicates whether the word was found and replaced.

    Parameters:
    - text (str): The text in which to replace the word.
    - word (str): The word to be replaced.
    - replacement (str): The word to replace the first occurrence of the target word with.

    Returns:
    - tuple:
        - str: The text with the first occurrence of the word replaced (if found).
        - bool: True if the word was found and replaced, False otherwise.
    """
    pattern = re.compile(r"\b" + word + r"\b", re.IGNORECASE)
    pattern2 = re.compile(word + r"\b", re.IGNORECASE)
    pattern3 = re.compile(r"\b" + word, re.IGNORECASE)
    modified_text, num_replacements = pattern.subn(replacement, text, count=1)

    if not num_replacements > 0:
        modified_text, num_replacements = pattern2.subn(replacement, text, count=1)
        if not num_replacements > 0:
            modified_text, num_replacements = pattern3.subn(replacement, text, count=1)

    word_found = num_replacements > 0
    return modified_text, word_found


def load_data(meta_file_path, mention_map_path):
    """
    Loads metadata and mention map from pickle files.

    Parameters:
    - meta_file_path (str): Path to the metadata pickle file.
    - mention_map_path (str): Path to the mention map pickle file.

    Returns:
    - dict: Metadata from the file.
    - dict: Original mention map from the file.
    """
    with open(meta_file_path, "rb") as f:
        meta_file = pickle.load(f)

    with open(mention_map_path, "rb") as f:
        original_mention_map = pickle.load(f)

    return meta_file, original_mention_map


def process_mentions(meta_file, original_mention_map, doc_sent_map):
    """
    Processes each sentence and mention from the loaded metadata and mention map,
    applying transformations as necessary.

    Parameters:
    - meta_file (dict): Metadata containing information about sentences and metaphors.
    - original_mention_map (dict): A mapping of mentions with their details.
    - doc_sent_map (dict): A mapping of doc id to their sentences
    """
    sentence_id2mentions = defaultdict(list)
    problematic_mids = []
    # Populate sentence_id2mentions
    for m_id, men in original_mention_map.items():
        if men["men_type"] == "evt" and men["split"] != "train":
            # sent_id = "_".join((men["doc_id"], men["sentence_id"]))
            sent_id = (men["doc_id"], men["sentence_id"])
            sentence_id2mentions[sent_id].append(m_id)

    # Process each sentence and mention
    for sent_id, m_ids in tqdm(
        list(sentence_id2mentions.items()), desc="Processing Meta Sentences"
    ):
        meta_dict = meta_file[sent_id]

        for m_id in m_ids:
            is_problematic = process_single_mention(
                m_id,
                meta_dict,
                original_mention_map,
                doc_sent_map[original_mention_map[m_id]["doc_id"]],
            )
            if is_problematic:
                problematic_mids.append(m_id)

    return problematic_mids


def process_single_mention(m_id, meta_dict, original_mention_map, sent_map):
    """
    Processes a single mention, updating its details based on the metaphor information.

    Parameters:
    - m_id (str): The mention ID.
    - meta_dict (dict): Metadata for the current sentence.
    - original_mention_map (dict): The original mention map.
    """
    curr_men = original_mention_map[m_id]
    if "Metaphoric Sentences" in meta_dict:
        meta_dict["Metaphoric Sentence"] = meta_dict["Metaphoric Sentences"]

    curr_men_txt = curr_men["mention_text"].strip()
    metaphor_sentence = meta_dict["Metaphoric Sentence"]
    if isinstance(metaphor_sentence, list):
        metaphor_sentence = metaphor_sentence[0]

    curr_men["sentence"] = meta_dict["Metaphoric Sentence"]
    sent_id = "_".join((curr_men["doc_id"], curr_men["sentence_id"]))

    ORIG_WORD_LIST = "Original Word List"
    META_WORD_LIST = "Metaphoric Word List"

    original_word_list = meta_dict[ORIG_WORD_LIST]
    if "Metaphoric Word List" in meta_dict:
        metaphoric_word_list = meta_dict[META_WORD_LIST]
    elif isinstance(original_word_list, dict):
        metaphoric_word_list = original_word_list

    if len(original_word_list) == 1 and isinstance(metaphoric_word_list, list):
        metaphoric_word_list = {}
        metaphoric_word_list[original_word_list[0]] = metaphoric_word_list
    elif isinstance(metaphoric_word_list, list):
        result = {}
        for d in metaphoric_word_list:
            result.update(d)
        metaphoric_word_list = result

    most_sim_key = find_most_similar(curr_men_txt, list(metaphoric_word_list.keys()))
    metaphor_words = metaphoric_word_list[most_sim_key]

    best_sim = 0.0
    best_match = ""
    for m_word in metaphor_words:
        if curr_men_txt.isupper():
            m_word = m_word.upper()
        match, sim = find_most_similar_complete_word(m_word, metaphor_sentence)
        if sim > best_sim:
            best_sim = sim
            best_match = match

    if best_sim == 0:
        metaphor_sentence = metaphor_sentence.lower()
        for m_word in metaphor_words:
            match, sim = find_most_similar_complete_word(
                m_word.lower(), metaphor_sentence
            )
            if sim > best_sim:
                best_sim = sim
                best_match = match

    if best_sim == 0:
        raise AssertionError()

    metaphor_sentence_marked, match_found = replace_first_occurrence(
        metaphor_sentence, best_match, f"<m> {best_match} </m>"
    )
    if not match_found:
        raise AssertionError

    curr_men["sentence"] = metaphor_sentence
    curr_men["marked_sentence"] = metaphor_sentence_marked
    curr_men["mention_text"] = best_match

    sent_map_new = copy.deepcopy(sent_map)
    sent_map_new[curr_men["sentence_id"]]["sentence"] = metaphor_sentence_marked
    curr_men["marked_doc"] = "\n".join(
        [sent["sentence"] for sent in sent_map_new.values()]
    )

    marked_doc = curr_men["marked_doc"]

    neighbors = marked_doc.split(metaphor_sentence_marked)
    neighbors_left = [
        sen.strip() for sen in neighbors[0].split("\n") if sen.strip() != ""
    ]
    neighbors_right = [
        sen.strip() for sen in neighbors[1].split("\n") if sen.strip() != ""
    ]

    curr_men["neighbors_left"] = neighbors_left
    curr_men["neighbors_right"] = neighbors_right

    if "<m>" not in metaphor_sentence_marked or "<m>" not in marked_doc:
        raise AssertionError

    return False


def add_lexical_features(mention_map):
    nlp = spacy.load("en_core_web_lg")
    mid_sents = [(val["sentence"], m_id) for m_id, val in mention_map.items()]

    mid_men_txt = [(val["mention_text"], m_id) for m_id, val in mention_map.items()]

    for doc, m_id in tqdm(
        nlp.pipe(mid_sents, as_tuples=True),
        desc="Adding Lexical Features",
        total=len(mid_sents),
    ):
        mention = mention_map[m_id]
        mention["sentence_tokens"] = [
            w.lemma_.lower()
            for w in doc
            if (not (w.is_stop or w.is_punct))
            or w.lemma_.lower() in {"he", "she", "his", "him", "her"}
        ]

    for doc, m_id in nlp.pipe(mid_men_txt, as_tuples=True):
        mention = mention_map[m_id]
        mention["lemma"] = " ".join(
            [
                w.lemma_.lower()
                for w in doc
                if (not (w.is_stop or w.is_punct))
                or w.lemma_.lower() in {"he", "she", "his", "him", "her"}
            ]
        )


@app.command()
def save_doc_sent_map(merged_file, original_doc_sent_map_file, output_file):
    meta_single = load_pickle(merged_file)
    doc_sent_map = load_pickle(original_doc_sent_map_file)

    process_meta_single(meta_single, doc_sent_map)

    save_pickle(doc_sent_map, output_file)


@app.command()
def parse(
    meta_file, doc_sent_map_file, original_mention_map_file, output_mention_map_file
):
    meta_file_single = load_pickle(meta_file)
    doc_sent_map = load_pickle(doc_sent_map_file)

    original_mention_map = load_pickle(original_mention_map_file)

    problematic_mids = process_mentions(
        meta_file_single, original_mention_map, doc_sent_map
    )

    print(len(problematic_mids))
    add_lexical_features(original_mention_map)
    pickle.dump(original_mention_map, open(output_mention_map_file, "wb"))
    print(f"Saved corpus file at {output_mention_map_file}")


if __name__ == "__main__":
    app()
