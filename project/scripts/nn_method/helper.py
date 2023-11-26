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
    msk_0 = q >= start_positions.unsqueeze(1)  # msk_0 marks the start of the mention (inclusive)
    msk_1 = q <= end_positions.unsqueeze(1)  # msk_1 marks the end of the mention (inclusive)
    attention_mask_g = (msk_0 & msk_1).int()  # attention_mask_g marks the tokens between the start and end of the mention (inclusive)

    msk_0_ar = q > start_positions.unsqueeze(1)  # msk_0_ar marks the tokens after the start of the mention (exclusive)
    msk_1_ar = q < end_positions.unsqueeze(1)  # msk_1_ar marks the tokens before the end of the mention (exclusive)
    arg = (msk_0_ar & msk_1_ar).int()  # arg marks the tokens before and after the mention (exclusive)

    return attention_mask_g, arg



def get_arg_attention_mask_wrapper(batch, m_start_id, m_end_id):
    input_ids = batch['input_ids']
    # get the attention mask for the arguments
    global_attention_mask, arg_attention_mask = get_arg_attention_mask(input_ids, m_start_id, m_end_id)
    batch['global_attention_mask'] = global_attention_mask
    batch['arg_attention_mask'] = arg_attention_mask
    return batch


def tokenize(
    tokenizer,
    mention_ids,
    mention_map,
    m_end,
    max_sentence_len=1024,
    text_key="bert_doc",
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
    instance_label_list = []
    instance_id_list = []
    doc_start, doc_end = "<doc-s>", "</doc-s>"

    for mention_id in mention_ids:
        datapoint = mention_map[mention_id]
        sentence = datapoint[text_key]
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
    tokenized_dict['label'] = instance_label_list
    tokenized_dict['mention_id'] = instance_id_list
    return tokenized_dict


def tokenize_with_postive_condiates(
    tokenizer,
    mention_ids,
    mention_map,
    m_end,
    max_sentence_len=1024,
    text_key="bert_doc",
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
    anchor_instance_list = []
    positive_candidate_instance_list = []
    anchor_gold_label_list = []
    positive_candidate_gold_label_list = []
    anchor_id_list = []
    doc_start, doc_end = "<doc-s>", "</doc-s>"
    

    for mention_id in mention_ids:
        anchor= mention_map[mention_id]
        anchor_sentence = anchor[text_key]
        anchor_instance = f"<g> {doc_start} {anchor_sentence} {doc_end}"
        anchor_instance_list.append(anchor_instance)
        anchor_gold_label_list.append(str(anchor[label_key]))
        anchor_id_list.append(mention_id)
        
        # Get the positive candidate
        positive_candidate = random.choice(mention_map[mention_id]['positive_candidates'])
        positive_candidate_sentence = positive_candidate[text_key]
        positive_candidate_instance = f"<g> {doc_start} {positive_candidate_sentence} {doc_end}"
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
        tokenized_anchor = tokenizer(anchor_instance_list, add_special_tokens=False, padding=True)
        tokenized_anchor_dict = {
            "input_ids": torch.LongTensor(tokenized_anchor["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized_anchor["attention_mask"]),
            "position_ids": torch.arange(tokenized_anchor["input_ids"].shape[-1]).expand(
                tokenized_anchor["input_ids"].shape
            ),
        }

        tokenized_positive = tokenizer(positive_candidate_instance_list, add_special_tokens=False, padding=True)
        tokenized_positive_dict = {
            "input_ids": torch.LongTensor(tokenized_positive["input_ids"]),
            "attention_mask": torch.LongTensor(tokenized_positive["attention_mask"]),
            "position_ids": torch.arange(tokenized_positive["input_ids"].shape[-1]).expand(
                tokenized_positive["input_ids"].shape
            ),
        }
        
    tokenized_anchor_dict['label'] = anchor_gold_label_list
    tokenized_positive_dict['label'] = positive_candidate_gold_label_list  
    tokenized_anchor_dict['mention_id'] = anchor_id_list
    return tokenized_anchor_dict, tokenized_positive_dict


#helper functions
@torch.no_grad()
def generate_embeddings(batch, model, device,):    
    input_ids = batch['input_ids'].to(device)
    attention_mask  = batch['attention_mask'].to(device)
    position_ids = batch['position_ids'].to(device)
    global_attention_mask = batch['global_attention_mask'].to(device)
    arg_attention_mask = batch['arg_attention_mask'].to(device)
    
    embeddings = model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                        global_attention_mask=global_attention_mask, arg=arg_attention_mask, )
    
    # move the embeddings to cpu to save gpu memory
    batch['embeddings'] = embeddings.cpu()
    return batch

    
def create_faiss_db(dataset, model, device):
    model = model.to(device)
    processed_dataset = dataset.map(lambda batch: generate_embeddings(batch, model, device), 
                                    batched=True, batch_size = 64)
    

    embeddings = processed_dataset['embeddings']
    hidden_size = embeddings.shape[-1]
    embeddings = embeddings.reshape(-1, hidden_size)
    # create the faiss index
    index = faiss.IndexFlatL2(hidden_size)
    index.add(embeddings)
    
    # clean cache
    
    return index, processed_dataset,

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
                if data_unit['label'] != true_label[index]:            
                    hard_negatives.append(data_unit)
                    break  # Break after finding the first hard negative
        
        # stack the list of dictionaries into a single dictionary
        input_id_list = torch.stack([data_unit['input_ids'] for data_unit in hard_negatives])
        attention_mask_list = torch.stack([data_unit['attention_mask'] for data_unit in hard_negatives])
        position_ids_list = torch.stack([data_unit['position_ids'] for data_unit in hard_negatives])
        global_attention_mask_list = torch.stack([data_unit['global_attention_mask'] for data_unit in hard_negatives])
        arg_attention_mask_list = torch.stack([data_unit['arg_attention_mask'] for data_unit in hard_negatives])
        return {'input_ids': input_id_list, 'attention_mask': attention_mask_list, 'position_ids': position_ids_list,
                'global_attention_mask': global_attention_mask_list, 'arg_attention_mask': arg_attention_mask_list}

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
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    position_ids = batch['position_ids'].to(device)
    global_attention_mask = batch['global_attention_mask'].to(device)
    arg_attention_mask = batch['arg_attention_mask'].to(device)
    
    embeddings = model(input_ids, attention_mask=attention_mask, position_ids=position_ids,
                       global_attention_mask=global_attention_mask, arg=arg_attention_mask)
    return embeddings