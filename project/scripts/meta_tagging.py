from collections import defaultdict
from prodigy.components.loaders import JSONL, JSON
from prodigy.components.preprocess import add_tokens
from prodigy.util import set_hashes
from typing import Optional

import copy
import json
import pickle
import prodigy
import spacy
from tqdm import tqdm

OLD_MARKED = """
<style>
    .myBox {
        font-size: 14px;
        border: none;
        padding: 20px;
        width: 100%;
        height: 100px;
        white-space: normal;
        overflow: scroll;
        line-height: 1.5;
    }
    
    .c0178m {
          box-sizing: border-box;
        }
        .c0179m {
          width: 100%;
          font-size: 20px;

        }
        .c0179m:focus {
          outline: 0;
        }
        .c0179m:empty {
          display: none;
        }
        .c0179m iframe, .c0179 img {
          maxwidth: 100%;
        }
        .c0180m {
          padding: 20px;
          padding: 20px;
          text-align: center;
        }
        .c01131m {
            border: 1px solid #ddd;
            text-align: left;
            border-radius: 4px;
        }
        .c01131m:focus-within {
          box-shadow: 0 0 0 1px #583fcf;
          border-color: #583fcf;
        }
        .c01132m {
          top: -3px;
          opacity: 0.75;
          position: relative;
          font-size: 12px;
          font-weight: bold;
          padding-left: 10px;
        }
        .c01133m {
          width: 100%;
          border: 0;
          padding: 10px;
          font-size: 20px;
          background: transparent;
          font-family: "Lato", "Trebuchet MS", Roboto, Helvetica, Arial, sans-serif;
        }
        .c01133m:focus {
          outline: 0;
        }  
        .prodigy-content{
            white-space: pre-wrap;
        }
</style>
<div class="c01131m">
<body>
<div class="myBox">
<p>
    {{{old_sentence}}}
</div>
</body>
</div>
<style onload="scrollToMark()" />
"""


def make_tasks(nlp, stream):
    """Add a 'spans' key to each example, with predicted entities."""
    # Process the stream using spaCy's nlp.pipe, which yields doc objects.
    # If as_tuples=True is set, you can pass in (text, context) tuples.
    texts = ((eg["text"], eg) for eg in stream)
    for doc, eg in nlp.pipe(texts, as_tuples=True):
        task = copy.deepcopy(eg)
        # Rehash the newly created task so that hashes reflect added data.
        task = set_hashes(
            task,
            input_keys=("text",),
            task_keys=("mention_id",),
            overwrite=True,
        )
        yield task


@prodigy.recipe(
    "meta-manual",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSON or JSONL file", "positional", None, str),
    port=("Port of the app", "option", "port", int),
)
def meta_manual(
    dataset: str,
    spacy_model: str,
    source: str,
    port: Optional[int] = 8080,
):
    nlp = spacy.load(spacy_model)
    labels = ["EVT"]
    stream = JSON(source)

    stream = add_tokens(nlp, stream)

    # Add the entities predicted by the model to the tasks in the stream.
    stream = make_tasks(nlp, stream)

    blocks = [
        {"view_id": "html", "html_template": OLD_MARKED},
        {"view_id": "ner_manual"},
        {
            "view_id": "text_input",
            "field_rows": 3,
            "field_autofocus": False,
            "field_label": "Reason for Flagging",
        },
    ]

    config = {
        "lang": nlp.lang,
        "labels": labels,  # Selectable label options
        "span_labels": labels,  # Selectable label options
        "auto_count_stream": True,  # Whether to recount the stream at initialization
        "show_stats": True,
        "host": "0.0.0.0",
        "port": port,
        "blocks": blocks,
    }

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "config": config,
    }


def make_json():
    nlp = spacy.load("en_core_web_sm")

    mention_map_old = pickle.load(open("../corpus/ecb/mention_map.pkl", "rb"))

    mention_map_meta = pickle.load(open("../corpus/ecb_meta_multi/mention_map.pkl", "rb"))

    meta2task = {}

    def process_sentence(sentence):
        # Remove spaces immediately after <m> and before </m>
        sentence = sentence.replace("<m> ", "<m>").replace(" </m>", "</m>")

        # Find the start index of the marked phrase
        start_index = sentence.find("<m>")

        # Remove the <m> and </m> tags
        sentence_without_markers = sentence.replace("<m>", "").replace("</m>", "")

        # Adjust the end index because of removed <m> tag
        end_index = sentence.find("</m>") - len("<m>")

        # Return the new sentence and the start and end indices of the marked phrase
        return sentence_without_markers, start_index, end_index

    tasks = []

    for m_id, meta_men in tqdm(list(mention_map_meta.items())):
        old_men = mention_map_old[m_id]
        marked_sentence = meta_men["marked_sentence"]
        clean_sentence, s, e = process_sentence(marked_sentence)

        old_sentence = old_men["marked_sentence"]
        old_sentence = old_sentence.replace("<m>", """<mark id="mark">""")
        old_sentence = old_sentence.replace("</m>", """</mark>""")

        task = {
            "mention_id": m_id,
            "old_sentence": old_sentence,
            "text": clean_sentence,
            "trigger": clean_sentence[s : e + 1],
        }
        doc = nlp(clean_sentence)
        span = doc.char_span(s, e)
        spans = []
        if span:
            spans.append(
                {
                    "token_start": span.start,
                    "token_end": span.end - 1,
                    "start": span.start_char,
                    "end": span.end_char,
                    "text": span.text,
                    "label": "EVT",
                }
            )
        task["spans"] = spans
        tasks.append(task)

    print(len(tasks))

    json.dump(tasks, open("../corpus/ecb_meta_multi/tasks.json", "w"), indent=1)


# make_json()