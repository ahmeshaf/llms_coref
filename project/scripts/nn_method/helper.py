"""
This module contains the helper functions for the BiEncoder, mostly for tokenization and attention masking.
"""

import random

import faiss
import torch
from torch.utils.data import Dataset as torch_Dataset


def get_arg_attention_mask(input_ids, m_start_id, m_end_id):
    """
    Get the global attention mask and the indices corresponding to the tokens between
    the mention indicators for sentences in a batch.

    Parameters
    ----------
    input_ids : Tensor
        A batch of sequences represented as IDs, with shape (batch_size, seq_len).
    encoder_model : nn.Module
        A model with attributes 'start_id' and 'end_id' indicating the special tokens' IDs.

    Returns
    -------
    attention_mask_g : Tensor
        The global attention masks for the sentences in the batch.
    arg : Tensor
        The masks for the argument/entity in the sentences in the batch.
    """
    input_ids = input_ids.cpu()
    batch_size, seq_len = input_ids.shape

    m_start_indicator = input_ids == m_start_id
    m_end_indicator = input_ids == m_end_id

    m = m_start_indicator + m_end_indicator
    nz_indexes = m.nonzero(as_tuple=True)
    batch_indices, token_indices = nz_indexes

    start_positions = torch.empty(batch_size, dtype=torch.long)
    end_positions = torch.empty(batch_size, dtype=torch.long)

    for i in range(batch_size):
        sequence_token_indices = token_indices[batch_indices == i]
        if sequence_token_indices.numel() != 2:
            raise ValueError(
                f"Sequence at index {i} must contain exactly two marker tokens."
            )
        start_positions[i], end_positions[i] = sequence_token_indices

    q = torch.arange(seq_len).repeat(batch_size, 1)
    msk_0 = q >= start_positions.unsqueeze(1)
    # msk_0 marks the start of the mention (inclusive)
    msk_1 = q <= end_positions.unsqueeze(1)
    # msk_1 marks the end of the mention (inclusive)
    attention_mask_g = (msk_0 & msk_1).int()
    # attention_mask_g marks the tokens between the start and end of the mention (inclusive)

    msk_0_ar = q > start_positions.unsqueeze(
        1
    )  # msk_0_ar marks the tokens after the start of the mention (exclusive)
    msk_1_ar = q < end_positions.unsqueeze(
        1
    )  # msk_1_ar marks the tokens before the end of the mention (exclusive)
    arg = (
        msk_0_ar & msk_1_ar
    ).int()  # arg marks the tokens before and after the mention (exclusive)

    return attention_mask_g, arg


def get_arg_attention_mask_wrapper(batch, m_start_id, m_end_id):
    input_ids = batch["input_ids"]
    # get the attention mask for the arguments
    global_attention_mask, arg_attention_mask = get_arg_attention_mask(
        input_ids, m_start_id, m_end_id
    )
    batch["global_attention_mask"] = global_attention_mask
    batch["arg_attention_mask"] = arg_attention_mask
    return batch


def tokenize_bi(
    tokenizer,
    mention_ids,
    mention_map,
    m_end,
    max_sentence_len=None,
    text_key="marked_doc",
    label_key="gold_cluster",
    truncate=True,
):
    """
    Tokenizes sentences, preserving special mention indicators and applying truncation if required.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer used for converting text to token IDs.
    mention_ids : list
        List of mention IDs to be tokenized.
    mention_map : dict
        A mapping from mention IDs to their corresponding sentences and other information.
    m_end : int
        The token ID for the end of a mention.
    max_sentence_len : int, optional
        The maximum length of the tokenized sentence. If not provided, defaults to the tokenizer's max length.
    text_key : str, optional
        The key in mention_map that holds the text to be tokenized. Defaults to 'marked_doc'.
    truncate : bool, optional
        Whether to apply truncation to the tokenized sentences. Defaults to True.

    Returns
    -------
    dict
        A dictionary containing the tokenized data with keys: 'input_ids', 'attention_mask', and 'position_ids'.
    """
    max_sentence_len = max_sentence_len or tokenizer.model_max_length
    instance_list = []
    instance_label_list = []
    instance_id_list = []
    doc_start, doc_end = "<doc-s>", "</doc-s>"

    for mention_id in mention_ids:
        datapoint = mention_map[mention_id]
        sentence = get_context(datapoint, text_key)
        instance = f"<g> {doc_start} {sentence} {doc_end}"
        instance_list.append(instance)
        instance_label_list.append(str(datapoint[label_key]))
        instance_id_list.append(mention_id)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)
            curr_start_index = max(0, m_end_index - (max_sentence_len // 2))
            in_truncated = (
                input_id[curr_start_index:m_end_index]
                + input_id[m_end_index : m_end_index + (max_sentence_len // 2)]
            )
            in_truncated += [tokenizer.pad_token_id] * (
                max_sentence_len - len(in_truncated)
            )
            input_ids_truncated.append(in_truncated)
        return torch.LongTensor(input_ids_truncated)

    def batch_tokenized(instance_list):
        tokenized = tokenizer(instance_list, add_special_tokens=False)
        tokenized_input_ids = (
            truncate_with_mentions(tokenized["input_ids"])
            if truncate
            else torch.LongTensor(tokenized["input_ids"])
        )
        positions_list = torch.arange(tokenized_input_ids.shape[-1]).expand(
            tokenized_input_ids.shape
        )
        attention_mask = tokenized_input_ids != tokenizer.pad_token_id
        return {
            "input_ids": tokenized_input_ids,
            "attention_mask": attention_mask,
            "position_ids": positions_list,
        }

    if truncate:
        tokenized_dict = batch_tokenized(instance_list)
    else:
        # If not truncating, tokenize the sentence pairs as single strings
        tokenized = tokenizer(instance_list, add_special_tokens=False, padding=True)
        tokenized_input_ids = torch.LongTensor(tokenized["input_ids"])
        tokenized_dict = {
            "input_ids": torch.LongTensor(tokenized["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized["attention_mask"]),
            "position_ids": torch.arange(tokenized_input_ids.shape[-1]).expand(
                tokenized_input_ids.shape
            ),
        }
    tokenized_dict["label"] = instance_label_list
    tokenized_dict["mention_id"] = instance_id_list
    return tokenized_dict


def tokenize_with_postive_condiates(
    tokenizer,
    mention_ids,
    mention_map,
    m_end,
    max_sentence_len=None,
    text_key="marked_doc",
    label_key="gold_cluster",
    truncate=True,
):
    """
    Tokenizes sentences along with their positive candidates, preserving special mention indicators and
    applying truncation if required.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer used for converting text to token IDs.
    mention_ids : List[int]
        List of mention IDs to be tokenized.
    mention_map : Dict[int, Dict[str, str]]
        A mapping from mention IDs to their corresponding sentences and other information.
    end_of_mention_token_id : int
        The token ID for the end of a mention.
    max_sentence_length : int, optional
        The maximum length of the tokenized sentence. Defaults to 1024.
    text_key : str, optional
        The key in mention_map that holds the text to be tokenized. Defaults to 'bert_doc'.
    apply_truncation : bool, optional
        Whether to apply truncation to the tokenized sentences. Defaults to True.

    Returns
    -------
    Tuple[Dict[str, torch.LongTensor], Dict[str, torch.LongTensor]]
        Two dictionaries containing the tokenized data for anchors and positive candidates respectively,
        with keys: 'input_ids', 'attention_mask', and 'position_ids'.
    """
    max_sentence_len = max_sentence_len or tokenizer.model_max_length
    print(max_sentence_len)
    anchor_instance_list = []
    positive_candidate_instance_list = []
    anchor_gold_label_list = []
    positive_candidate_gold_label_list = []
    anchor_id_list = []
    doc_start, doc_end = "<doc-s>", "</doc-s>"

    for mention_id in mention_ids:
        anchor = mention_map[mention_id]
        anchor_sentence = sentence = get_context(anchor, text_key)
        anchor_instance = f"<g> {doc_start} {anchor_sentence} {doc_end}"
        anchor_instance_list.append(anchor_instance)
        anchor_gold_label_list.append(str(anchor[label_key]))
        anchor_id_list.append(mention_id)

        # Get the positive candidate
        positive_candidate = mention_map[
            random.choice(mention_map[mention_id]["positive_candidates"])
        ]
        positive_candidate_sentence = get_context(positive_candidate, text_key)
        positive_candidate_instance = (
            f"<g> {doc_start} {positive_candidate_sentence} {doc_end}"
        )
        positive_candidate_instance_list.append(positive_candidate_instance)
        positive_candidate_gold_label_list.append(str(positive_candidate[label_key]))

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)
            curr_start_index = max(0, m_end_index - (max_sentence_len // 2))
            in_truncated = (
                input_id[curr_start_index:m_end_index]
                + input_id[m_end_index : m_end_index + (max_sentence_len // 2)]
            )
            in_truncated += [tokenizer.pad_token_id] * (
                max_sentence_len - len(in_truncated)
            )
            input_ids_truncated.append(in_truncated)
        return torch.LongTensor(input_ids_truncated)

    def batch_tokenized(instance_list):
        tokenized = tokenizer(instance_list, add_special_tokens=False)
        tokenized_input_ids = (
            truncate_with_mentions(tokenized["input_ids"])
            if truncate
            else torch.LongTensor(tokenized["input_ids"])
        )
        positions_list = torch.arange(tokenized_input_ids.shape[-1]).expand(
            tokenized_input_ids.shape
        )
        attention_mask = tokenized_input_ids != tokenizer.pad_token_id
        return {
            "input_ids": tokenized_input_ids,
            "attention_mask": attention_mask,
            "position_ids": positions_list,
        }

    if truncate:
        tokenized_anchor_dict = batch_tokenized(anchor_instance_list)
        tokenized_positive_dict = batch_tokenized(positive_candidate_instance_list)

    else:
        # Tokenize the sentence pairs as single strings if not truncating
        tokenized_anchor = tokenizer(
            anchor_instance_list, add_special_tokens=False, padding=True
        )
        tokenized_anchor_dict = {
            "input_ids": torch.LongTensor(tokenized_anchor["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized_anchor["attention_mask"]),
            "position_ids": torch.arange(
                tokenized_anchor["input_ids"].shape[-1]
            ).expand(tokenized_anchor["input_ids"].shape),
        }

        tokenized_positive = tokenizer(
            positive_candidate_instance_list, add_special_tokens=False, padding=True
        )
        tokenized_positive_dict = {
            "input_ids": torch.LongTensor(tokenized_positive["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized_positive["attention_mask"]),
            "position_ids": torch.arange(
                tokenized_positive["input_ids"].shape[-1]
            ).expand(tokenized_positive["input_ids"].shape),
        }

    tokenized_anchor_dict["label"] = anchor_gold_label_list
    tokenized_positive_dict["label"] = positive_candidate_gold_label_list
    tokenized_anchor_dict["mention_id"] = anchor_id_list
    return tokenized_anchor_dict, tokenized_positive_dict


def get_arg_attention_mask_ce(input_ids, parallel_model):
    """
    Get the global attention mask and the indices corresponding to the tokens between
    the mention indicators.
    Parameters
    ----------
    input_ids
    parallel_model

    Returns
    -------
    Tensor, Tensor, Tensor
        The global attention mask, arg1 indicator, and arg2 indicator
    """
    input_ids.cpu()

    num_inputs = input_ids.shape[0]

    m_start_indicator = input_ids == parallel_model.module.start_id
    m_end_indicator = input_ids == parallel_model.module.end_id

    m = m_start_indicator + m_end_indicator

    # non-zero indices are the tokens corresponding to <m> and </m>
    nz_indexes = m.nonzero()[:, 1].reshape((num_inputs, 4))

    # Now we need to make the tokens between <m> and </m> to be non-zero
    q = torch.arange(m.shape[1])
    q = q.repeat(m.shape[0], 1)

    # all indices greater than and equal to the first <m> become True
    msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the first </m> become True
    msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
    # all indices greater than and equal to the second <m> become True
    msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the second </m> become True
    msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

    # excluding <m> and </m> gives only the indices between <m> and </m>
    msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
    msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

    # Union of indices between first <m> and </m> and second <m> and </m>
    attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()
    # attention_mask_g = None
    # attention_mask_g[:, 0] = 1

    # indices between <m> and </m> excluding the <m> and </m>
    arg1 = msk_0_ar.int() * msk_1_ar.int()
    arg2 = msk_2_ar.int() * msk_3_ar.int()

    return attention_mask_g, arg1, arg2


def forward_ab(parallel_model, ab_dict, device, indices, lm_only=False):
    batch_tensor_ab = ab_dict["input_ids"][indices, :]
    batch_am_ab = ab_dict["attention_mask"][indices, :]
    batch_posits_ab = ab_dict["position_ids"][indices, :]
    am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask_ce(
        batch_tensor_ab, parallel_model
    )

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)
    if am_g_ab is not None:
        am_g_ab.to(device)
    arg1_ab.to(device)
    arg2_ab.to(device)

    return parallel_model(
        batch_tensor_ab,
        attention_mask=batch_am_ab,
        position_ids=batch_posits_ab,
        global_attention_mask=am_g_ab,
        arg1=arg1_ab,
        arg2=arg2_ab,
        lm_only=lm_only,
    )


def get_context(m_dict, text_key):
    if text_key in {"marked_sentence", "marked_doc"}:
        return m_dict[text_key]
    elif text_key.startswith("neighbors"):
        neighbor_count = int(text_key.split("_")[-1])
        neighbors_left = m_dict["neighbors_left"]
        neighbors_right = m_dict["neighbors_right"]
        sent_neighbors = (
            neighbors_left[:neighbor_count]
            + [m_dict["marked_sentence"]]
            + neighbors_right[:neighbor_count]
        )
        return "\n".join(sent_neighbors)
    else:
        raise ValueError(
            "Invalid text_key. It should be either marked_doc, marked_sentence or of the form neighbors_n"
        )


def tokenize_ce(
    tokenizer,
    mention_pairs,
    mention_map,
    m_end,
    max_sentence_len=None,
    text_key="bert_doc",
    truncate=True,
):
    if max_sentence_len is None:
        max_sentence_len = tokenizer.model_max_length

    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    doc_start = "<doc-s>"
    doc_end = "</doc-s>"

    for m1, m2 in mention_pairs:
        sentence_a = get_context(mention_map[m1], text_key)
        sentence_b = get_context(mention_map[m2], text_key)

        def make_instance(sent_a, sent_b):
            return " ".join(["<g>", doc_start, sent_a, doc_end]), " ".join(
                [doc_start, sent_b, doc_end]
            )

        instance_ab = make_instance(sentence_a, sentence_b)
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)

            curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

            in_truncated = (
                input_id[curr_start_index:m_end_index]
                + input_id[m_end_index : m_end_index + (max_sentence_len // 4)]
            )
            in_truncated = in_truncated + [tokenizer.pad_token_id] * (
                max_sentence_len // 2 - len(in_truncated)
            )
            input_ids_truncated.append(in_truncated)

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances):
        instances_a, instances_b = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a["input_ids"])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b["input_ids"])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))

        tokenized_ab_dict = {
            "input_ids": tokenized_ab_,
            "attention_mask": (tokenized_ab_ != tokenizer.pad_token_id),
            "position_ids": positions_ab,
        }

        return tokenized_ab_dict

    if truncate:
        tokenized_ab = ab_tokenized(pairwise_bert_instances_ab)
        tokenized_ba = ab_tokenized(pairwise_bert_instances_ba)
    else:
        instances_ab = [" ".join(instance) for instance in pairwise_bert_instances_ab]
        instances_ba = [" ".join(instance) for instance in pairwise_bert_instances_ba]
        tokenized_ab = tokenizer(
            list(instances_ab), add_special_tokens=False, padding=True
        )

        tokenized_ab_input_ids = torch.LongTensor(tokenized_ab["input_ids"])

        tokenized_ab = {
            "input_ids": torch.LongTensor(tokenized_ab["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized_ab["attention_mask"]),
            "position_ids": torch.arange(tokenized_ab_input_ids.shape[-1]).expand(
                tokenized_ab_input_ids.shape
            ),
        }

        tokenized_ba = tokenizer(
            list(instances_ba), add_special_tokens=False, padding=True
        )
        tokenized_ba_input_ids = torch.LongTensor(tokenized_ba["input_ids"])
        tokenized_ba = {
            "input_ids": torch.LongTensor(tokenized_ba["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized_ba["attention_mask"]),
            "position_ids": torch.arange(tokenized_ba_input_ids.shape[-1]).expand(
                tokenized_ba_input_ids.shape
            ),
        }

    return tokenized_ab, tokenized_ba


# helper functions
@torch.no_grad()
def generate_embeddings(
    batch,
    model,
    device,
):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    position_ids = batch["position_ids"].to(device)
    global_attention_mask = batch["global_attention_mask"].to(device)
    arg_attention_mask = batch["arg_attention_mask"].to(device)

    embeddings = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        global_attention_mask=global_attention_mask,
        arg=arg_attention_mask,
    )

    # move the embeddings to cpu to save gpu memory
    batch["embeddings"] = embeddings.cpu()
    return batch


def create_faiss_db(dataset, model, device):
    model = model.to(device)
    processed_dataset = dataset.map(
        lambda batch: generate_embeddings(batch, model, device),
        batched=True,
        batch_size=64,
    )

    embeddings = processed_dataset["embeddings"]
    hidden_size = embeddings.shape[-1]
    embeddings = embeddings.reshape(-1, hidden_size).cpu().numpy()
    # create the faiss index
    index = faiss.IndexFlatL2(hidden_size)
    index.add(embeddings)

    # clean cache

    return (
        index,
        processed_dataset,
    )


class VectorDatabase:
    def __init__(self):
        self.faiss_index = None
        self.dataset = None

    def set_index(self, faiss_index, dataset):
        self.faiss_index = faiss_index
        self.dataset = dataset

    def get_hard_negative(self, anchor_embeddings, true_label):
        # Convert anchor embeddings from tensor to numpy for FAISS
        anchor_embeddings_np = anchor_embeddings.detach().cpu().numpy()
        # Search the index for the 50 nearest neighbors of the anchor embeddings
        _, nearest_neighbors_list = self.faiss_index.search(anchor_embeddings_np, 50)
        hard_negatives = []
        for index, neighbors in enumerate(nearest_neighbors_list):
            # Iterate over each set of neighbors to find a hard negative case
            # print(nearest_neighbors_list)
            for neighbor_idx in neighbors:
                data_unit = self.dataset[int(neighbor_idx)]
                if data_unit["label"] != true_label[index]:
                    hard_negatives.append(data_unit)
                    break  # Break after finding the first hard negative

        # stack the list of dictionaries into a single dictionary
        input_id_list = torch.stack(
            [data_unit["input_ids"] for data_unit in hard_negatives]
        )
        attention_mask_list = torch.stack(
            [data_unit["attention_mask"] for data_unit in hard_negatives]
        )
        position_ids_list = torch.stack(
            [data_unit["position_ids"] for data_unit in hard_negatives]
        )
        global_attention_mask_list = torch.stack(
            [data_unit["global_attention_mask"] for data_unit in hard_negatives]
        )
        arg_attention_mask_list = torch.stack(
            [data_unit["arg_attention_mask"] for data_unit in hard_negatives]
        )
        return {
            "input_ids": input_id_list,
            "attention_mask": attention_mask_list,
            "position_ids": position_ids_list,
            "global_attention_mask": global_attention_mask_list,
            "arg_attention_mask": arg_attention_mask_list,
        }

    def get_nearest_neighbors(self, anchor_embeddings, k=10):
        """
        Retrieve the k nearest neighbors for each anchor embedding.

        :param anchor_embeddings: Tensor of anchor embeddings.
        :param k: Number of nearest neighbors to retrieve.
        :return: A list of k nearest neighbors for each anchor.
        """
        # Convert anchor embeddings from tensor to numpy for FAISS
        anchor_embeddings_np = anchor_embeddings.detach().cpu().numpy()
        # Search the index for the k nearest neighbors of the anchor embeddings
        _, nearest_neighbors_indices = self.faiss_index.search(anchor_embeddings_np, k)

        nearest_neighbors = []
        for neighbors in nearest_neighbors_indices:
            neighbor_data_units = []
            for neighbor_idx in neighbors:
                data_unit = self.dataset[int(neighbor_idx)]
                neighbor_data_units.append(data_unit)
            nearest_neighbors.append(neighbor_data_units)

        return nearest_neighbors


class CombinedDataset(torch_Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        item1 = self.dataset1[idx]
        item2 = self.dataset2[idx]
        return item1, item2


def process_batch(batch, model, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    position_ids = batch["position_ids"].to(device)
    global_attention_mask = batch["global_attention_mask"].to(device)
    arg_attention_mask = batch["arg_attention_mask"].to(device)

    embeddings = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        global_attention_mask=global_attention_mask,
        arg=arg_attention_mask,
    )
    return embeddings
