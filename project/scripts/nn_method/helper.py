"""
This module contains the helper functions for the BiEncoder, mostly for tokenization and attention masking.
"""

import torch


def get_arg_attention_mask(input_ids, encoder_model):
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

    m_start_indicator = input_ids == encoder_model.module.start_id
    m_end_indicator = input_ids == encoder_model.module.end_id

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
    msk_0 = q >= start_positions.unsqueeze(1)  # msk_0 marks the start of the mention (inclusive)
    msk_1 = q <= end_positions.unsqueeze(1)  # msk_1 marks the end of the mention (inclusive)
    attention_mask_g = (msk_0 & msk_1).int()  # attention_mask_g marks the tokens between the start and end of the mention (inclusive)

    msk_0_ar = q > start_positions.unsqueeze(1)  # msk_0_ar marks the tokens after the start of the mention (exclusive)
    msk_1_ar = q < end_positions.unsqueeze(1)  # msk_1_ar marks the tokens before the end of the mention (exclusive)
    arg = (msk_0_ar & msk_1_ar).int()  # arg marks the tokens before and after the mention (exclusive)

    return attention_mask_g, arg


def tokenize(
    tokenizer,
    mention_ids,
    mention_map,
    m_end,
    max_sentence_len=1024,
    text_key="bert_doc",
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
        The key in mention_map that holds the text to be tokenized. Defaults to 'bert_doc'.
    truncate : bool, optional
        Whether to apply truncation to the tokenized sentences. Defaults to True.

    Returns
    -------
    dict
        A dictionary containing the tokenized data with keys: 'input_ids', 'attention_mask', and 'position_ids'.
    """
    max_sentence_len = max_sentence_len or tokenizer.model_max_length
    instance_list = []
    doc_start, doc_end = "<doc-s>", "</doc-s>"

    for mention_id in mention_ids:
        sentence = mention_map[mention_id][text_key]
        instance = f"<g> {doc_start} {sentence} {doc_end}"
        instance_list.append(instance)

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
    return tokenized_dict
