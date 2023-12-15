import json
import os
import pickle
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openai
import typer
from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser
from tqdm import tqdm

from .bert.helper import get_context
from .coref_prompt_collections import (
    baseline_output_parser,
    baseline_prompt,
    cot_output_parser,
    eightshot_prompt,
    fourshot_prompt,
    twoshot_prompt,
    zeroshot_prompt,
)
from .helper import ensure_dir, ensure_path, evaluate
from .special_prompt_collections import (
    adv_prompt,
    adv_prompt_v2,
    amr_prompt,
    arg_prompt,
    davidson_prompt,
    quine_prompt,
    tag_prompt,
    summarization_prompt,
)

app = typer.Typer()
random.seed(42)


class JsonParser:
    def __init__(self):
        pass

    def parse(self, output):
        return json.loads(output)


def prompt_and_parser_factory(
    prompt_type: str,
) -> Tuple[PromptTemplate, StructuredOutputParser]:
    if prompt_type == "baseline":
        return baseline_prompt, baseline_output_parser
    elif prompt_type == "zeroshot":
        return zeroshot_prompt, cot_output_parser
    elif prompt_type == "twoshot":
        return twoshot_prompt, cot_output_parser
    elif prompt_type == "fourshot":
        return fourshot_prompt, cot_output_parser
    elif prompt_type == "eightshot":
        return eightshot_prompt, cot_output_parser
    elif prompt_type == "davidson":
        return davidson_prompt, JsonParser()
    elif prompt_type == "quine":
        return quine_prompt, JsonParser()
    elif prompt_type == "amr":
        return amr_prompt, JsonParser()
    elif prompt_type == "arg":
        return arg_prompt, JsonParser()
    elif prompt_type == "adv":
        return adv_prompt, JsonParser()
    elif prompt_type == "adv_v2":
        return adv_prompt_v2, JsonParser()
    elif prompt_type == "sum":
        return summarization_prompt, JsonParser()
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")


def extract_answers(strings):
    # Regular expression to extract the "Answer" part
    pattern = r'"Answer":\s*"(\w+)"'
    answers = []
    matches = re.findall(pattern, strings)
    answers.extend(matches)
    return answers


def extract_answers_beta(strings):
    # Regular expression to extract the "Answer" part
    pattern1 = r'"*Answer"*\s*:\s*(true|false)'
    pattern2 = r"is\s*(true|false)"

    # First, try to find matches with pattern1
    matches = re.findall(pattern1, strings, re.IGNORECASE)
    if not matches:
        # If no matches found with pattern1, try with pattern2
        matches = re.findall(pattern2, strings, re.IGNORECASE)
    # Extracting the true|false part
    answers = [match for match in matches if match]
    return answers


def process_response(tuple_id, response):
    try:
        lines = response["predict"].strip().split("\n")
        if "}" not in lines[-1]:
            lines.append("}")
        structured_lines = []
        for line in lines:
            line = line.strip('", ')
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
                value = value.strip(' ",')
                value = f"{json.dumps(value)},"
                structured_lines.append(key + ":" + value)
            else:
                print("Unknown line: ", line)

        structured_lines[-2] = structured_lines[-2].strip(",")
        structured_json = "\n".join(structured_lines)
        structured_json = json.loads(structured_json)
        return structured_json
    except Exception as e:
        print(f"An error occurred at {id}:", e)
        return None


def llm_coref(
    event_pairs: List[Tuple[str, str]],
    mention_map: Dict,
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    cache_dir: Path = "/tmp/gpt_pred_coref/",
    gpt_version: str = "gpt-4",
    save_folder: Path = "../../llm_results",
    text_key: str = "marked_doc",
    run_name: str = "baseline",
    temperature: float = 0.7,
) -> Dict:
    """
    Predict coreference using the LLM (Language Model) for provided event pairs.

    Parameters
    ----------
    event_pairs : list of tuple of str
        A list of event pairs where each pair is represented as a tuple of event IDs.
    mention_map : dict
        A dictionary mapping event IDs to their respective data.
        Data should include 'bert_sentence' representing the event's sentence.
    prompt : PromptTemplate
        A template for the LLM prompt to query the model.
    parser : StructuredOutputParser
        A parser to extract structured output from the LLM's response.
    gpt_version : str, optional
        The version of the GPT model to use. Default is "gpt-4".
    cache_file: Path
    text_key: str
    run_name: str
    save_folder : str, optional
        Folder to save the prediction results. Default is "../../llm_results".
    temperature: float


    Returns
    -------
    list of int
        A list of prediction results, where 1 represents 'True' and 0 represents 'False'.
    dict
        A dictionary containing detailed prediction results for each event pair,
        including event IDs, event sentences, and parsed predictions.
    """
    cache_file = os.path.join(cache_dir, f"{gpt_version}_{run_name}_cache.pkl")
    ensure_path(cache_file)
    cache_file = Path(cache_file)

    if cache_file.exists():
        raw_cache = pickle.load(open(cache_file, "rb"))
    else:
        raw_cache = {}

    result_list = []
    result_dict = defaultdict(dict)
    # result_dict = not_found.add((m1, m2))

    # initialize the llm
    llm = ChatOpenAI(
        temperature=temperature, model=gpt_version, request_timeout=180
    )  # Set the request_timeout to 180 seconds
    chain = LLMChain(llm=llm, prompt=prompt)

    # predict
    for evt_pair in tqdm(event_pairs):
        event1_id = evt_pair[0]
        event2_id = evt_pair[1]

        event1_data = mention_map.get(event1_id)
        event2_data = mention_map.get(event2_id)

        if event1_data is None or event2_data is None:
            continue

        event1_text = get_context(event1_data, text_key)
        event2_text = get_context(event2_data, text_key)

        # format_prompt = chain.prompt.format_prompt(event1=event1_text, event2=event2_text)

        if evt_pair in raw_cache:
            predict = raw_cache[evt_pair]["predict"]
        else:
            with get_openai_callback() as cb:
                predict = chain.run(event1=event1_text, event2=event2_text)
                predict_cost = {
                    "Total": cb.total_tokens,
                    "Prompt": cb.prompt_tokens,
                    "Completion": cb.completion_tokens,
                    "Cost": cb.total_cost,
                }
                raw_cache[evt_pair] = {"predict": predict, "predict_cost": predict_cost}
                pickle.dump(raw_cache, open(cache_file, "wb"))
        try:
            predict_dict = parser.parse(predict)

        except Exception as e:
            print(e)
            # llm might occasionally generate multiple predictions
            # in this case, we take the first one, following the setting in the paper cot.
            answers = extract_answers(str(predict))
            print("total answers detected: ", len(answers))
            if answers:
                predict_dict = {"Answer": answers[0]}
            else:
                print("No answer detected, set to False")
                predict_dict = {"Answer": "False"}

        result_list.append(predict_dict["Answer"])
        result_dict[evt_pair] = {
            "event1_id": event1_id,
            "event2_id": event2_id,
            "event1": event1_text,
            "event2": event2_text,
            "event1_trigger": event1_data["mention_text"],
            "event2_trigger": event2_data["mention_text"],
            "predict_raw": predict,
            "predict_dict": predict_dict,
        }

    # Save the result_dict
    save_folder = os.path.join(save_folder, run_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        save_folder,
        f"{gpt_version}_{run_name}_predict_result_{timestamp}.pkl",
    )
    # pickle.dump(result_dict, open(save_path, "wb"))
    # print(f"Saved prediction results to {save_path}")

    result_array = []
    str_count = 0
    bool_count = 0
    other_count = 0
    for r in result_list:
        if isinstance(r, str):
            if r == "True":
                r = 1
            else:
                r = 0
            str_count += 1
        elif isinstance(r, bool):
            if r:
                r = 1
            else:
                r = 0
            bool_count += 1
        else:
            print("Unknown type: ", type(r))
            print("r: ", r)
            r = 0
            other_count += 1
        result_array.append(r)
    print("Total: ", len(result_list))
    print("str_count: ", str_count)
    print("bool_count: ", bool_count)
    print("other_count: ", other_count)
    result_array = np.array(result_array)
    return result_array, result_dict


def llm_aug(
    flatten_doc_sent_map: Dict,
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    cache_dir: Path = "/tmp/gpt_argu/",
    gpt_version: str = "gpt-4",
    save_folder: Path = "../../llm_argu",
    run_name: str = "adv",
    temperature: float = 0.7,
) -> Dict:
    # Prepare the cache file
    cache_file = os.path.join(cache_dir, f"{gpt_version}_{run_name}_cache.pkl")
    ensure_path(cache_file)
    cache_file = Path(cache_file)

    if cache_file.exists():
        raw_cache = pickle.load(open(cache_file, "rb"))
    else:
        raw_cache = {}

    # Initialize the result dict
    result_dict = defaultdict(dict)

    # initialize the llm
    llm = ChatOpenAI(
        temperature=temperature, model=gpt_version, request_timeout=180
    )  # Set the request_timeout to 180 seconds
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate
    for key, value in tqdm(flatten_doc_sent_map.items()):
        if value is None:
            continue

        format_prompt = chain.prompt.format_prompt(
            sentence=value["sentence"], mention_texts=",".join(value["mention_texts"])
        )
        if key in raw_cache:
            predict = raw_cache[key]["predict"]
        else:
            with get_openai_callback() as cb:
                predict = chain.run(
                    sentence=value["sentence"],
                    mention_texts=", ".join(value["mention_texts"]),
                )

                predict_cost = {
                    "Total": cb.total_tokens,
                    "Prompt": cb.prompt_tokens,
                    "Completion": cb.completion_tokens,
                    "Cost": cb.total_cost,
                }
                raw_cache[key] = {
                    "predict": predict,
                    "predict_cost": predict_cost,
                    "format_prompt": format_prompt,
                }
                pickle.dump(raw_cache, open(cache_file, "wb"))
        try:
            predict_dict = parser.parse(predict)
            predict_dict["format_prompt"] = format_prompt

        except Exception as e:
            print(e)

        result_dict[key] = predict_dict

    # Save the result_dict
    save_folder = os.path.join(save_folder, run_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        save_folder,
        f"{gpt_version}_{run_name}_argument_result_{timestamp}.pkl",
    )
    pickle.dump(result_dict, open(save_path, "wb"))
    print(f"Saved prediction results to {save_path}")

    return result_dict


def llm_tagging(
    doc_id2m_ids: Dict,
    mention_map: Dict,
    aug_paraphrases_json: Dict,
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    cache_dir: Path = "/tmp/gpt_tagging/",
    gpt_version: str = "gpt-4",
    save_folder: Path = "../../llm_tagging",
    run_name: str = "tagging",
    temperature: float = 0.7,
) -> Dict:
    # Prepare the cache file
    cache_file = os.path.join(cache_dir, f"{gpt_version}_{run_name}_cache.pkl")
    ensure_path(cache_file)
    cache_file = Path(cache_file)

    if cache_file.exists():
        raw_cache = pickle.load(open(cache_file, "rb"))
    else:
        raw_cache = {}

    # Initialize the result dict
    result_dict = defaultdict(dict)

    # initialize the llm
    llm = ChatOpenAI(
        temperature=temperature, model=gpt_version, request_timeout=180
    )  # Set the request_timeout to 180 seconds
    chain = LLMChain(llm=llm, prompt=prompt)

    paraphrase_key = ["Most Esoteric", "Moderately Esoteric", "Less Esoteric"]

    # Generate
    for doc_id, m_ids in tqdm(doc_id2m_ids.items()):
        curr_key = random.choice(paraphrase_key)
        for m_id in m_ids:
            sentence_id = str(
                mention_map[m_id]["sentence_id"]
            )  # Use mention_id to get the sentence_id
            if (doc_id, sentence_id) in aug_paraphrases_json:
                paraphrase_sentence = aug_paraphrases_json[(doc_id, sentence_id)][
                    curr_key
                ]
                marked_sentence = mention_map[m_id]["marked_sentence"]
                event_trigger = mention_map[m_id]["mention_text"]

                input_dict = {
                    "original_sentence": marked_sentence,
                    "original_event_trigger": event_trigger,
                }

                if m_id in raw_cache:
                    predict = raw_cache[m_id]["predict"]
                else:
                    with get_openai_callback() as cb:
                        predict = chain.run(
                            original_sentence=marked_sentence,
                            original_event_trigger=event_trigger,
                            paraphrased_sentence=paraphrase_sentence,
                        )
                        predict_cost = {
                            "Total": cb.total_tokens,
                            "Prompt": cb.prompt_tokens,
                            "Completion": cb.completion_tokens,
                            "Cost": cb.total_cost,
                        }

                        raw_cache[m_id] = {
                            "predict": predict,
                            "predict_cost": predict_cost,
                            "input": input_dict,
                            "paraphrase_key": curr_key,
                        }
                        pickle.dump(raw_cache, open(cache_file, "wb"))
                try:
                    predict_dict = parser.parse(predict)
                    predict_dict["input"] = input_dict
                    result_dict[m_id] = predict_dict
                except Exception as e:
                    print(e)

    # Save the result_dict
    save_folder = os.path.join(save_folder, run_name)
    ensure_dir(save_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        save_folder,
        f"{gpt_version}_{run_name}_tagging_result_{timestamp}.pkl",
    )
    pickle.dump(result_dict, open(save_path, "wb"))
    print(f"Saved prediction results to {save_path}")

    return result_dict


def llm_sum(
    mention_map: Dict,
    split_mention_ids: List[str],
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    cache_dir: Path = "/tmp/gpt_sum/",
    gpt_version: str = "gpt-4",
    save_folder: Path = "../../llm_sum",
    run_name: str = "tagging",
    temperature: float = 0.7,
):
    # Prepare the cache file
    cache_file = os.path.join(cache_dir, f"{gpt_version}_{run_name}_cache.pkl")
    ensure_path(cache_file)
    cache_file = Path(cache_file)

    if cache_file.exists():
        raw_cache = pickle.load(open(cache_file, "rb"))
    else:
        raw_cache = {}

    # Initialize the result dict
    result_dict = defaultdict(dict)

    # initialize the llm
    llm = ChatOpenAI(
        temperature=temperature, model=gpt_version, request_timeout=180
    )  # Set the request_timeout to 180 seconds
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate
    for m_id in tqdm(split_mention_ids):
        marked_sentence = mention_map[m_id]["marked_sentence"]
        marked_doc = mention_map[m_id]["marked_doc"]
        doc = marked_doc.replace("<m>", "").replace("</m>", "")
        format_prompt = chain.prompt.format_prompt(
            marked_sentence=marked_sentence, document=doc
        )
        if m_id in raw_cache:
            predict = raw_cache[m_id]["predict"]
        else:
            with get_openai_callback() as cb:
                predict = chain.run(
                    marked_sentence=marked_sentence,
                    document=doc,
                )
                predict_cost = {
                    "Total": cb.total_tokens,
                    "Prompt": cb.prompt_tokens,
                    "Completion": cb.completion_tokens,
                    "Cost": cb.total_cost,
                }

                raw_cache[m_id] = {
                    "predict": predict,
                    "predict_cost": predict_cost,
                    "format_prompt": format_prompt,
                }
                pickle.dump(raw_cache, open(cache_file, "wb"))
        try:
            predict_dict = parser.parse(predict)
            predict_dict["format_prompt"] = format_prompt
            result_dict[m_id] = predict_dict
        except Exception as e:
            print(e)
    # Save the result_dict
    save_folder = os.path.join(save_folder, run_name)
    ensure_dir(save_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        save_folder,
        f"{gpt_version}_{run_name}_sum_result_{timestamp}.pkl",
    )
    pickle.dump(result_dict, open(save_path, "wb"))
    print(f"Saved prediction results to {save_path}")

    return result_dict


def llm_two_shot(
    event_pairs: List[Tuple[str, str]],
    mention_map: Dict,
    m2two_shot: Path,
    prompt: PromptTemplate,
    parser: StructuredOutputParser,
    cache_dir: Path = "/tmp/gpt_pred_coref/",
    gpt_version: str = "gpt-4",
    save_folder: Path = "../../llm_results",
    text_key: str = "marked_doc",
    run_name: str = "two_shot",
    temperature: float = 0.7,
):
    cache_file = os.path.join(cache_dir, f"{gpt_version}_{run_name}_cache.pkl")
    ensure_path(cache_file)
    cache_file = Path(cache_file)

    if cache_file.exists():
        raw_cache = pickle.load(open(cache_file, "rb"))
    else:
        raw_cache = {}

    result_list = []
    result_dict = defaultdict(dict)

    # initialize the llm
    llm = ChatOpenAI(
        temperature=temperature, model=gpt_version, request_timeout=180
    )  # Set the request_timeout to 180 seconds
    chain = LLMChain(llm=llm, prompt=prompt)

    m2two_shot_map = pickle.load(open(m2two_shot, "rb"))

    # predict
    for evt_pair in tqdm(event_pairs):
        if evt_pair not in m2two_shot_map:
            raise ValueError(
                "pair not found in the two-shot map. Try running the two-shot-getter on these pairs first"
            )
        two_shots = m2two_shot_map[evt_pair]
        pos_pair = two_shots["pos"]
        neg_pair = two_shots["neg"]

        coreferent1 = get_context(mention_map[pos_pair[0]], text_key)
        coreferent2 = get_context(mention_map[pos_pair[1]], text_key)
        non_coreferent1 = get_context(mention_map[neg_pair[0]], text_key)
        non_coreferent2 = get_context(mention_map[neg_pair[1]], text_key)

        event1_id = evt_pair[0]
        event2_id = evt_pair[1]

        event1_data = mention_map.get(event1_id)
        event2_data = mention_map.get(event2_id)

        if event1_data is None or event2_data is None:
            continue

        event1_text = get_context(event1_data, text_key)
        event2_text = get_context(event2_data, text_key)

        # format_prompt = chain.prompt.format_prompt(event1=event1_text, event2=event2_text)

        if evt_pair in raw_cache:
            predict = raw_cache[evt_pair]["predict"]
        else:
            with get_openai_callback() as cb:
                # TODO: get the coreferent1, coreferent2, non-corefrent1, non-coreferent2
                # predict = chain.run(event1=event1_text, event2=event2_text,
                #                     coreferent1=coreferent1, coreferent2=coreferent2,
                #                     non-coreferent1=non-coreferent1, non-coreferent2=non-coreferent2)
                predict = chain.run(
                    event1=event1_text,
                    event2=event2_text,
                    coreferent1=coreferent1,
                    coreferent2=coreferent2,
                    non_coreferent1=non_coreferent1,
                    non_coreferent2=non_coreferent2,
                )
                predict_cost = {
                    "Total": cb.total_tokens,
                    "Prompt": cb.prompt_tokens,
                    "Completion": cb.completion_tokens,
                    "Cost": cb.total_cost,
                }
                raw_cache[evt_pair] = {"predict": predict, "predict_cost": predict_cost}
                pickle.dump(raw_cache, open(cache_file, "wb"))
        try:
            predict_dict = parser.parse(predict)
            answer = predict_dict["Answer"]
        except Exception as e:
            print("An error occurred at ", evt_pair, ": ", e)
            print("Using self-defined parsing function")
            predict_dict = process_response(evt_pair, raw_cache[evt_pair])

            if predict_dict is None:
                # Extract the answer from the raw response
                answers = extract_answers_beta(str(predict))
                if len(answers) == 0:
                    print("Edge Case Countered:", evt_pair)
                    predict_dict = {"Answer": "False"}

                elif len(answers) > 1:
                    print("taking the first answer")
                    predict_dict = {"Answer": answers[0]}

                else:
                    predict_dict = {"Answer": answers}

        result_list.append(predict_dict["Answer"])
        result_dict[evt_pair] = {
            "event1_id": event1_id,
            "event2_id": event2_id,
            "event1": event1_text,
            "event2": event2_text,
            "event1_trigger": event1_data["mention_text"],
            "event2_trigger": event2_data["mention_text"],
            "predict_raw": predict,
            "predict_dict": predict_dict,
        }

    # Save the result_dict
    save_folder = os.path.join(save_folder, run_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        save_folder,
        f"{gpt_version}_{run_name}_predict_result_{timestamp}.pkl",
    )
    # pickle.dump(result_dict, open(save_path, "wb"))
    # print(f"Saved prediction results to {save_path}")

    result_array = []
    str_count = 0
    bool_count = 0
    other_count = 0
    for r in result_list:
        if isinstance(r, str):
            if r.lower() == "true":  # to make sure the evaluation more reasonable
                r = 1
            else:
                r = 0
            str_count += 1
        elif isinstance(r, bool):
            if r:
                r = 1
            else:
                r = 0
            bool_count += 1
        else:
            print("Unknown type: ", type(r))
            print("r: ", r)
            r = 0
            other_count += 1
        result_array.append(r)
    print("Total: ", len(result_list))
    print("str_count: ", str_count)
    print("bool_count: ", bool_count)
    print("other_count: ", other_count)
    result_array = np.array(result_array)
    return result_array, result_dict


@app.command()
def run_llm_two_shot(
    dataset_folder: str,
    split: str,
    mention_pairs_path: Path,
    m2two_shot: Path,
    results_file: Path,
    debug: bool = False,
    gpt_version: str = "gpt-4",
    gpt_raw_cache_dir: Path = "/tmp/gpt_pred_coref/",
    save_folder: Path = "../../llm_results",
    experiment_name: str = "baseline",
    text_key: str = "marked_doc",
):
    m2two_shot = pickle.load(open(m2two_shot, "rb"))
    mention_pairs = pickle.load(open(mention_pairs_path, "rb"))

    """
        p1 = mention_pairs[0]
        m2two_shot[p1]
        { "pos": (m_pos1, m_pos2), "neg": (m_neg1, m_neg2)}
        
        Here's a similar coreferent pair:
        Coreferent 1: mention_map[m_pos1][marked_sentence]
        Coreferent 2: mention_map[m_pos2][marked_sentence]
        
        Here's a similar non-coreferent pair:
        Non-coreferent 1: mention_map[m_neg1][marked_sentence]
        Non-coreferent 2: mention_map[m_neg2][marked_sentence]
        
        Your task is to deduce why or how Coreferent 1 and 2 are the same. And, Non-coreferent 1 and 2 are not the same.
        
        Using your deductive reasoning, identify if Event 1 and Event 2 below are coreferent or not.
        Answer in JSON Format as:
        {
            Coreferent 1 and 2 rationale: ""
            Non-coreferent 1 and 2 rationale: ""
            
            Event 1 Trigger: Text in Event 1 between <m> and </m>
            Event 2 Trigger: Text in Event 2 between <m> and </m>
            
            Event 1 and 2 deduction: Your explanation for the coreference or non-coreference relation between Event 1 and Event 2
            Answer: True if Event 1 and 2 are the same, else False
        }
        
        Now lets solve
        Event 1 = { event_1 }
        Event 2 = { event_2 }
    """
    ensure_dir(gpt_raw_cache_dir)
    # Set up openai api key
    _ = load_dotenv(find_dotenv())  # Read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Read the mention_map from the dataset_folder
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))

    split_mention_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == split
    ]

    # Get the mention pairs
    mention_pairs = sorted(pickle.load(open(mention_pairs_path, "rb")))
    mention_pairs = [tuple(sorted(p)) for p in mention_pairs]

    # filtering
    print("before filtering: ", len(mention_pairs))

    mention_pairs = filter_topic_aligned_cross_doc_pairs(mention_pairs, mention_map)
    print("after filtering: ", len(mention_pairs))

    # debug
    if debug:
        print("Total mention ids: ", len(split_mention_ids))
        print("Total mention pairs: ", len(mention_pairs))
        mention_pairs = mention_pairs[:5]

    prompt, parser = prompt_and_parser_factory(experiment_name)

    result_list, result_dict = llm_two_shot(
        mention_pairs,
        mention_map,
        m2two_shot,
        prompt,
        parser,
        gpt_version=gpt_version,
        save_folder=save_folder,
        run_name=experiment_name,
        text_key=text_key,
        cache_dir=gpt_raw_cache_dir,
    )

    # Evaluate the result
    result_array = np.array(result_list)

    scores = evaluate(
        mention_map, split_mention_ids, mention_pairs, similarity_matrix=result_array
    )

    # Save the results
    pickle.dump(
        (mention_pairs, result_list, result_array),
        open(results_file, "wb"),
    )

    # Debug
    if debug:
        print(result_dict)

    print(scores)


@app.command()
def run_llm_pipeline(
    dataset_folder: str,
    split: str,
    mention_pairs_path: Path,
    results_file: Path,
    debug: bool = False,
    gpt_version: str = "gpt-4",
    gpt_raw_cache_dir: Path = "/tmp/gpt_pred_coref/",
    save_folder: Path = "../../llm_results",
    experiment_name: str = "baseline",
    text_key: str = "marked_doc",
):
    ensure_dir(gpt_raw_cache_dir)
    # Set up openai api key
    _ = load_dotenv(find_dotenv())  # Read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Read the mention_map from the dataset_folder
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))

    split_mention_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == split
    ]

    # Get the mention pairs
    mention_pairs = sorted(pickle.load(open(mention_pairs_path, "rb")))
    mention_pairs = [tuple(sorted(p)) for p in mention_pairs]

    # debug
    if debug:
        print("Total mention ids: ", len(split_mention_ids))
        print("Total mention pairs: ", len(mention_pairs))
        mention_pairs = mention_pairs[:5]

    prompt, parser = prompt_and_parser_factory(experiment_name)

    result_list, result_dict = llm_coref(
        mention_pairs,
        mention_map,
        prompt,
        parser,
        gpt_version=gpt_version,
        save_folder=save_folder,
        run_name=experiment_name,
        text_key=text_key,
        cache_dir=gpt_raw_cache_dir,
    )

    # Evaluate the result
    result_array = np.array(result_list)

    scores = evaluate(
        mention_map, split_mention_ids, mention_pairs, similarity_matrix=result_array
    )

    # Save the results
    pickle.dump(
        (mention_pairs, result_list, result_array),
        open(results_file, "wb"),
    )

    # Debug
    if debug:
        print(result_dict)

    print(scores)


@app.command()
def run_llm_aug_pipeline(
    dataset_folder: str,
    split: str,
    debug: bool = False,
    gpt_version: str = "gpt-4",
    gpt_raw_cache_dir: Path = "/tmp/gpt_aug/",
    save_folder: Path = "../../llm_argu_results",
    experiment_name: str = "adv",
):
    ensure_path(gpt_raw_cache_dir)
    # Set up openai api key
    _ = load_dotenv(find_dotenv())  # Read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Read the mention_map from the dataset_folder
    doc_sent_map = pickle.load(open(dataset_folder + "/doc_sent_map.pkl", "rb"))

    # Flatten the doc_sent_map
    flattened_dict = defaultdict(dict)
    for doc_key, sentences in doc_sent_map.items():
        for sent_key, sent_data in sentences.items():
            new_key = (doc_key, sent_data["sent_id"])
            flattened_dict[new_key] = sent_data["sentence"]

    prompt, parser = prompt_and_parser_factory(experiment_name)

    # TODO: remove below line when done
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    flattened_dict = defaultdict(dict)
    # get the debug split
    for key, value in mention_map.items():
        if value["split"] == split and value["men_type"] == "evt":
            flattened_dict[(value["doc_id"], value["sentence_id"])]["sentence"] = value[
                "sentence"
            ]
            if (
                flattened_dict[(value["doc_id"], value["sentence_id"])].get(
                    "mention_texts"
                )
                is None
            ):
                flattened_dict[(value["doc_id"], value["sentence_id"])][
                    "mention_texts"
                ] = [value["mention_text"]]
            else:
                flattened_dict[(value["doc_id"], value["sentence_id"])][
                    "mention_texts"
                ].append(value["mention_text"])

    # debug
    if debug:
        print("Total sentences: ", len(flattened_dict))
        first_5_items = list(flattened_dict.items())[:5]
        flattened_dict = dict(first_5_items)

    # breakpoint()
    result_dict = llm_aug(
        flattened_dict,
        prompt,
        parser,
        gpt_version=gpt_version,
        save_folder=save_folder,
        run_name=experiment_name,
        cache_dir=gpt_raw_cache_dir,
    )
    # Debug
    if debug:
        print(result_dict)


@app.command()
def run_llm_tagging_pipeline(
    aug_cache_file: Path,
    dataset_folder: str,
    split: str,
    debug: bool = False,
    gpt_version: str = "gpt-4",
    gpt_raw_cache_dir: Path = "/tmp/gpt_tagging/",
    save_folder: Path = "../../llm_tagging_results",
):
    # Set up openai api key
    _ = load_dotenv(find_dotenv())  # Read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))

    aug_paraphrases = pickle.load(open(aug_cache_file, "rb"))
    aug_paraphrases_json = {}
    for tuple_id, response in tqdm(aug_paraphrases.items()):
        try:
            structured_json = json.loads(response["predict"])
        except Exception as e:
            structured_json = process_response(tuple_id, response)
            if structured_json is None:
                continue
        aug_paraphrases_json[tuple_id] = structured_json

    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))

    doc_id2m_ids = defaultdict(list)
    for m_id, mention in mention_map.items():
        if (
            mention["men_type"] == "evt" and mention["split"] == split
        ):  # TODO: remove this line when done
            doc_id2m_ids[mention["doc_id"]].append(m_id)

    # debug
    if debug:
        print("Total docs: ", len(doc_id2m_ids))
        first_5_items = list(doc_id2m_ids.items())[:5]
        doc_id2m_ids = dict(first_5_items)

    result_dict = llm_tagging(
        doc_id2m_ids,
        mention_map,
        aug_paraphrases_json,
        tag_prompt,
        JsonParser(),
        gpt_version=gpt_version,
        save_folder=save_folder,
        cache_dir=gpt_raw_cache_dir,
    )

    if debug:
        print(result_dict)


@app.command()
def run_llm_sum_pipeline(
    dataset_folder: str,
    split: str,
    debug: bool = False,
    gpt_version: str = "gpt-4",
    gpt_raw_cache_dir: Path = "/tmp/gpt_sum/",
    save_folder: Path = "../../llm_sum_results",
    experiment_name: str = "sum",
):
    ensure_dir(gpt_raw_cache_dir)
    # Set up openai api key
    _ = load_dotenv(find_dotenv())  # Read local .env file
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Read the mention_map from the dataset_folder
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    split_mention_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == split
    ]

    if debug:
        print("Total mention ids: ", len(split_mention_ids))
        split_mention_ids = split_mention_ids[:5]

    prompt, parser = prompt_and_parser_factory(experiment_name)

    result_dict = llm_sum(
        mention_map,
        split_mention_ids,
        prompt,
        parser,
        gpt_version=gpt_version,
        save_folder=save_folder,
        run_name=experiment_name,
        cache_dir=gpt_raw_cache_dir,
    )

    if debug:
        print(result_dict)


if __name__ == "__main__":
    app()
