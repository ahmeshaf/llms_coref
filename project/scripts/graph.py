"""
Generate contextualized embeddings for the given data
Reference:
https://huggingface.co/sentence-transformers/all-mpnet-base-v2
"""
import os
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_embeddings(data_dict, model_name, save_path):
    # Initialize the model
    model = SentenceTransformer(model_name)

    # Prepare batch processing
    spans = []
    sentences = []
    for key in sorted(data_dict.keys()):
        value = data_dict[key]
        spans.append(value["mention_text"])
        sentences.append(value["sentence"])

    # Generate embeddings in batches
    span_embeddings = model.encode(spans, batch_size=32, show_progress_bar=True)
    sentence_embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Save the embeddings
    torch.save(span_embeddings, os.path.join(save_path, "span_embeddings.pt"))
    torch.save(sentence_embeddings, os.path.join(save_path, "sentence_embeddings.pt"))

    print("Embeddings generated and saved successfully.")


if __name__ == "__main__":
    # Load the data
    with open("data/mention_map_evt_multi.pkl", "rb") as f:
        data_dict = pickle.load(f)

    # Debug
    # sample_dict = {key: value for key, value in list(data_dict.items())}
    
    # Generate embeddings
    generate_embeddings(data_dict, "all-mpnet-base-v2", "embeddings/meta_multi/")