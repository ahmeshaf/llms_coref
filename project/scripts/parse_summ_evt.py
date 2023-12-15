import pickle
# from llm_pipeline import process_response
import json
import re
from collections import Counter

import spacy


mention_map = pickle.load(open("../corpus/ecb_meta/mention_map.pkl", "rb"))
#
for mention in mention_map.values():
    marked_sentence = mention["marked_sentence"]
    m = "<m>"
    m_e = "</m>"
    assert m in marked_sentence
    assert m_e in marked_sentence
    assert len(marked_sentence.split(m)) == 2
    assert len(marked_sentence.split(m_e)) == 2


def process_response(id_, response):
    try:
        lines = response.strip().split("\n")
        if "}" not in lines[-1]:
            lines.append("}")
        structured_lines = []
        for i, line in enumerate(lines):
            line = line.strip('", ')
            if line.strip() == "}":
                structured_lines[i-1] = structured_lines[i-1].strip(",")
                if i != len(line) - 1:
                    structured_lines.append("},")
                    continue
            if line == "":
                continue
            if len(line.split(":")) == 1:
                structured_lines.append(line)
            elif len(line.split(":")) >= 2:
                splits = line.split(":")
                key = splits[0]
                value = ":".join(splits[1:])

                key = key.strip(' ",')
                key = f"{json.dumps(key)}"
                if value.strip() != "{":
                    value = value.strip(' ",')
                    value = f"{json.dumps(value)},"
                structured_lines.append(key + ":" + value)
            else:
                print("Unknown line: ", line)

        structured_lines[-2] = structured_lines[-2].strip(",")
        structured_json = "\n".join(structured_lines)
        structured_json = json.loads(structured_json.strip(","))
        return structured_json
    except AttributeError as e:
        print(f"An error occurred at {id_}:", e)
        return None


summ_cache_file = pickle.load(open("../corpus/ecb_sum_gpt4/gpt-4-1106-preview_sum_cache_full.pkl", "rb"))

mention_map = pickle.load(open("../corpus/ecb/mention_map.pkl", "rb"))

nlp = spacy.load("en_core_web_lg")
print(len(summ_cache_file))
for m_id, response in summ_cache_file.items():
    mention = mention_map[m_id]
    predict_ = response["predict"].strip()
    if "```json" in predict_:
        pattern = r'```json(.*?)```'

        matches = re.findall(pattern, predict_, re.DOTALL)[0]
        predict_ = matches.strip()
    try:
        json_res = json.loads(predict_)
        summary = json_res["Event Summary in One Sentence with Marked Trigger"]
    except Exception as e:
        lines = predict_.split("\n")
        summary = ""
        for line in lines:
            if "Event Summary in One Sentence with Marked Trigger" in line:
                summary = line.split(":")[-1].strip(' ",')

    if summary == "":
        raise Exception
    # print(summary)
    clean_sentence = summary.replace("<m>", "").replace("<m>", "").replace("</m>", "").replace("</m>", "")
    clean_sentence.replace("/wiki/", " /wiki/ ")
    mention["marked_sentence"] = mention["marked_sentence"] + " <s> " + clean_sentence

    # mention["sentence"] = marked_sentence
    # mention["mention_text"] = trigger
    # mention["lemma"] = nlp(trigger)[:].root.lemma_
    # mention["sentence_tokens"] = [t.lemma_ for t in nlp(marked_sentence)]
    # marked_sentence = str(marked_sentence).replace("[", "").replace("]", "")
    # if len(marked_sentence.split("<m>")) > 2:
    #     print(marked_sentence)
    # if "<m>" in marked_sentence and "</m>" in marked_sentence:
    #     mention["marked_sentence"] = marked_sentence
    # elif "<m>" in marked_sentence and "</m>" not in marked_sentence:
    #     print(marked_sentence)
    # else:
    #     # print(marked_sentence)
    #     if re.search(trigger, marked_sentence, flags=re.IGNORECASE):
    #         # print(trigger)
    #         # print(marked_sentence)
    #         marked_sentence = re.sub(trigger, f"<m> {trigger} </m> ", marked_sentence, flags=re.IGNORECASE, count=1)
    #     elif re.search(mention["lemma"], marked_sentence, flags=re.IGNORECASE):
    #         marked_sentence = re.sub(marked_sentence, mention["lemma"], f"<m> {trigger} </m> ", flags=re.IGNORECASE)
    #     else:
    #         print("no match found")
    #         # print(trigger)
    #         marked_sentence = f"<m> {trigger} </m> " + marked_sentence
    #         # marked_sentence = "On Thursday, Dr Mark Vinar <m> disappeared </m> off Zurbriggens Ridge, on Mt Cook."
    #         # print(marked_sentence)
    #
    #     mention["marked_sentence"] = marked_sentence
    #
    # print(marked_sentence)

pickle.dump(mention_map, open("../corpus/ecb_sum_gpt4/mention_map.pkl", "wb"))

for mention in mention_map.values():
    marked_sentence = mention["marked_sentence"]
    m = "<m>"
    m_e = "</m>"
    assert m in marked_sentence
    assert m_e in marked_sentence
    assert len(marked_sentence.split(m)) == 2
    assert len(marked_sentence.split(m_e)) == 2