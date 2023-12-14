# Read the paraphrased doc and tagged triggers
from pathlib import Path

from typer import Typer

import pickle

app = Typer()

""""
'32_3ecb.xml', '0') 
{'predict': '{\n   
"Less Esoteric" :  "The police and prosecutors have stated that the New Bedford man, who was taken into custody last night, committed the tragic act of taking the lives of his own mother and ex-girlfriend.",\n   
"Moderately Esoteric" :  "Under the watchful eyes of law enforcement and legal representatives, the New Bedford man, apprehended during the previous evening, perpetrated the heart-wrenching act of extinguishing the lives of his maternal figure and former romantic partner.",\n   "Most Esoteric" :  "Amidst the judicial authorities and legal advocates, the New Bedford man, apprehended under the cloak of darkness, perpetrated the grievous act of terminating the existence of his progenitor and erstwhile paramour."\n}',
"""


def parse_ecb_meta(
    original_mention_map_file: Path,
    og_doc_sent_map: Path,
    paraphrased_json_file: Path,
    tagging_json_file: Path,
    output_mention_map_file: Path,
):
    # load orig, para, tag pickle files
    og_mention_map, og_doc_sent_map, paraphrased_map, tag_map = None

    out_mention_map = {}

    ## TODO extract esoteric level for each doc_id from tag_map. You may need to use og_mention_map
    doc2estoric_map = {"doc_id": "Least Esoteric"}
    for (doc_id, sentence_id) in paraphrased_json_file.items():
        pass
        # change each sentence in og_doc_sent_map
        og_doc_sent_map[]


    for m_id, tag_dict in tag_map.items():
        pass
        # TODO: extract the sent_id = (doc_id, sentence_id)
        # get the paraphrased sentence from paraphrased_map[sent_id]
        # get the tag_dict["Event Trigger"]
        # get the tag_dict["Marked Sentence"]
        og_mention = og_mention_map[m_id]
        # assign the correct keys in og_mention (marked_sentence, sentence, mention_text, lemma, marked_doc)

        # use spacy to get the head lemma of the mention text. use nlp(mention_text).root.lemma_
        out_mention_map[m_id] = og_mention

    pickle.dump(out_mention_map, open(output_mention_map_file, "wb"))

