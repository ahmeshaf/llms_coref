import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import spacy
from typer import Typer

global nlp
nlp = spacy.load("en_core_web_lg")

app = Typer()


# Function to process and structure the JSON response
def structure_json_response(
    response_id: str, response_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    try:
        response_lines = response_data["predict"].strip().split("\n")

        # Ensure the JSON string is properly closed
        if "}" not in response_lines[-1]:
            response_lines.append("}")

        structured_response = []
        for line in response_lines:
            line = line.strip('", ')
            if line == "":
                continue

            line_parts = line.split(":")
            if len(line_parts) == 1:
                structured_response.append(line)
            elif len(line_parts) >= 2:
                key, value = line_parts[0], ":".join(line_parts[1:])
                key = json.dumps(key.strip(' ",'))
                value = json.dumps(value.strip(' ",')) + ","
                structured_response.append(f"{key}:{value}")
            else:
                print("Unknown line format: ", line)

        structured_response[-2] = structured_response[-2].rstrip(",")
        structured_json_string = "\n".join(structured_response)
        return json.loads(structured_json_string)
    except Exception as e:
        print(f"An error occurred processing response ID {response_id}:", e)
        return None


# Process and parse the tagging map
def parse_tagging_map(tagging_map: Dict[str, Dict[str, Any]]) -> None:
    for response_id, response_data in tagging_map.items():
        try:
            tagging_map[response_id]["predict"] = json.loads(response_data["predict"])
        except Exception as e:
            parsed_response = structure_json_response(response_id, response_data)
            if parsed_response is not None:
                print(
                    "Parsed response ID with Customized parsing function:", response_id
                )
                tagging_map[response_id]["predict"] = parsed_response
            else:
                print("Failed to parse response ID:", response_id)
                print("Response data:")
                print(response_data)

    print("Parsing complete.")


@app.command()
def parse_ecb_meta(
    original_mention_map_file: Path,
    original_doc_sent_map: Path,
    paraphrased_pickle_file: Path,
    tagging_pickle_file: Path,
    output_mention_map_file: Path,
):
    # load original mention map
    og_mention_map = pickle.load(open(original_mention_map_file, "rb"))
    og_doc_sent_map = pickle.load(open(original_doc_sent_map, "rb"))
    paraphrased_map = pickle.load(open(paraphrased_pickle_file, "rb"))
    tagging_map = pickle.load(open(tagging_pickle_file, "rb"))

    parse_tagging_map(tagging_map)
    parse_tagging_map(paraphrased_map)

    out_mention_map = {}

    # Construct the doc2esoteric_map
    doc2esoteric_map = {}
    for m_id, tag_dict in tagging_map.items():
        doc_id = og_mention_map[m_id]["doc_id"]
        esoteric_level = tag_dict["paraphrase_key"]
        doc2esoteric_map[doc_id] = esoteric_level

    # Construct the doc2paraphrase_map
    doc2paraphrase_map = defaultdict(dict)
    for tuple_id, p_dict in paraphrased_map.items():
        esoteric_level = doc2esoteric_map[doc_id]
        doc_id, sent_id = tuple_id
        doc2paraphrase_map[doc_id][sent_id] = p_dict["predict"][esoteric_level]

    # Order the paraphrased sentences by sentence ID
    for doc_id, sent_dict in doc2paraphrase_map.items():
        sorted_sent_dict = {k: sent_dict[k] for k in sorted(sent_dict)}
        doc2paraphrase_map[doc_id] = sorted_sent_dict

    for m_id, tag_dict in tagging_map.items():
        og_mention = og_mention_map[m_id]
        sent_id = og_mention["sentence_id"]
        doc_id = og_mention["doc_id"]
        tagging_triggers = tag_dict["predict"]["Paraphrased Trigger"]
        tagging_marked_sentence = tag_dict["predict"]["Paraphrased Marked Sentence"]
        tagged_trigger_lemma = nlp(tagging_triggers)[:].root.lemma_
        paraphrased_marked_doc = " ".join(
            [value for key, value in doc2paraphrase_map[doc_id].items()]
        )

        og_mention["marked_sentence"] = tagging_marked_sentence
        og_mention["marked_doc"] = paraphrased_marked_doc
        og_mention["lemma"] = tagged_trigger_lemma
        og_mention["mention_text"] = tagging_triggers
        out_mention_map[m_id] = og_mention

    pickle.dump(out_mention_map, open(output_mention_map_file, "wb"))

    print("Saved to:", output_mention_map_file)


import pickle
import re
from pathlib import Path


def check_format(sentence):
    condition1 = "<m>" in sentence
    condition2 = "</m>" in sentence
    condition3 = len(sentence.split("<m>")) == 2
    condition4 = len(sentence.split("</m>")) == 2
    return condition1 and condition2 and condition3 and condition4


@app.command()
def check_ecb_meta(mention_map_file: Path, save_file: Path):
    with open(mention_map_file, "rb") as file:
        mention_map = pickle.load(file)

    error_count = 0
    correction_count = 0
    final_error = 0
    attention_list = []

    for key, mention in mention_map.items():
        marked_sentence = mention["marked_sentence"]
        trigger = mention["mention_text"]
        m_start, m_end = "<m>", "</m>"

        if not check_format(marked_sentence):
            error_count += 1

            # Attempt to correct the marked sentence
            cleaned_sentence = marked_sentence.replace("<m>", "").replace("</m>", "")
            cleaned_trigger = trigger.replace("<m>", "").replace("</m>", "")
            pattern = re.compile(re.escape(cleaned_trigger), re.IGNORECASE)
            tagged_sentence = re.sub(
                pattern, lambda m: f"<m>{m.group(0)}</m>", cleaned_sentence, count=1
            )

            if not check_format(tagged_sentence):
                # add the trigger to the end of the sentence
                tagged_sentence = f"{tagged_sentence} <m>{cleaned_trigger}</m>"

                final_error += 1
                print(f"Trigger: {trigger}")
                print(f"Cleaned:")
                print(cleaned_sentence)
                print(f"Marked:")
                print(marked_sentence)
                print(f"Tagged:")
                print(tagged_sentence)
                print("=========")

                if not check_format(tagged_sentence):
                    attention_list.append(
                        (key, marked_sentence, tagged_sentence, trigger)
                    )

            mention_map[key]["marked_sentence"] = tagged_sentence
            print("changed marked sentence")

    print(
        f"Finished checking {len(mention_map)} mentions with {error_count} initial errors."
    )
    print(f"Corrections made: {correction_count}")
    print(f"Final error count (uncorrected): {final_error}")
    print(f"Attention list: {attention_list}")

    # Save the mention map
    pickle.dump(mention_map, open(save_file, "wb"))

    print("Saved to:", save_file)


if __name__ == "__main__":
    app()
