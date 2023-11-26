import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from bi_encoder import BiEncoder
from helper import (CombinedDataset, VectorDatabase, create_faiss_db,
                    get_arg_attention_mask_wrapper, process_batch, tokenize,
                    tokenize_with_postive_condiates)


def evaluate(model, mention_dict, selected_keys, device, top_k=10):

    # check the model is on the specified device
    model.to(device)
    tokenizer = model.tokenizer
    m_start_id = model.start_id
    m_end_id = model.end_id

    # tokenize the dev set
    tokenized_dev_dict = tokenize(tokenizer, selected_keys, mention_dict, m_end_id)
    dev_dataset = Dataset.from_dict(tokenized_dev_dict).with_format("torch")
    dev_dataset = dev_dataset.map(
        lambda batch: get_arg_attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True, batch_size=1
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    
    faiss_db = VectorDatabase()
    dev_index, _ = create_faiss_db(dev_dataset, model, device=device)
    faiss_db.set_index(dev_index, dev_dataset)
    
    true_positives = 0
    total = 0
    singleton_count = 0
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            # skip the batch if the mention is a singleton
            mention_id = batch['mention_id'][0]
            if mention_dict[mention_id]['tag_descriptor'] == 'singleton':
                singleton_count += 1
                continue
            
            embeddings = process_batch(batch, model, device)
            # Retrieve the nearest neighbors
            neighbors_list = faiss_db.get_nearest_neighbors(embeddings, top_k+1)
            # Remove the first neighbor, which is the mention itself
            neighbors_list = neighbors_list[0][1:]
            # Check if the correct mention is in the neighbors
            gold_label = batch['label'][0]
            neighbor_label_list = [neighbor['label'] for neighbor in neighbors_list]
            correct_mention_found = gold_label in neighbor_label_list
            if correct_mention_found:
                true_positives += 1
            total += 1
            
    print("true_positives:", true_positives)
    print("total:", total)
    recall = true_positives / total if total > 0 else 0
    print("recall:", recall)


def train(mention_dict, selected_keys, 
                           dev_mention_dict, dev_selected_keys, 
                           batch_size=2, epochs=1):
    # Initialize the model and tokenizer
    bi_encoder = BiEncoder()
    tokenizer = bi_encoder.tokenizer
    m_start_id = bi_encoder.start_id
    m_end_id = bi_encoder.end_id

    # Tokenize anchors and positive candidates
    tokenized_anchor_dict, tokenized_positive_dict = tokenize_with_postive_condiates(
        tokenizer, selected_keys, mention_dict, m_end_id
    )

    # Prepare datasets
    train_dataset = Dataset.from_dict(tokenized_anchor_dict).with_format("torch")
    train_dataset = train_dataset.map(
        lambda batch: get_arg_attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True, batch_size=batch_size
    )

    train_dataset_positive_candidates = Dataset.from_dict(tokenized_positive_dict).with_format("torch")
    train_dataset_positive_candidates = train_dataset_positive_candidates.map(
        lambda batch: get_arg_attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True, batch_size=batch_size
    )

    # DataLoaders
    combined_dataset = CombinedDataset(train_dataset, train_dataset_positive_candidates)
    train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bi_encoder.to(device)

    # FAISS database setup
    faiss_db = VectorDatabase()
    train_index, _ = create_faiss_db(train_dataset, model, device=device)
    faiss_db.set_index(train_index, train_dataset)

    # Loss function and optimizer
    margin = 10.0
    triplet_loss = lambda anchors, positives, negatives, margin: F.triplet_margin_loss(
            anchors,
            positives,
            negatives,
            margin=margin,
            p=2,
            eps=1e-6,
            swap=False,
            reduction="mean",
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # start the evaluation
    evaluate(model, dev_mention_dict, dev_selected_keys, device)

    # Training loop
    model.train()
    for epoch in range(epochs):
        # Initialize progress bar
        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
        
        for i, (anchor_batch, positive_batch) in enumerate(train_dataloader, start=1):
            # Process anchor, positive, and negative batches
            anchor_embeddings = process_batch(anchor_batch, model, device)
            positive_embeddings = process_batch(positive_batch, model, device)
            
            hard_negatives_batch = faiss_db.get_hard_negative(anchor_embeddings, anchor_batch['label'])
            negative_embeddings = process_batch(hard_negatives_batch, model, device)
        
            # Compute loss and update model
            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.update()
            pbar.set_postfix({'Iteration': f'{i}/{len(train_dataloader)}', 'Loss': f'{loss.item():.4f}'})
            
        pbar.close()
        # Save the model
        torch.save(model.state_dict(), f"/home/zhiyong/Projects/llms_coref/project/bi_encoder_{epoch}.pt")
        
        # Update the FAISS database
        try:
            train_index, _ = create_faiss_db(train_dataset, model, device=device)
            faiss_db.set_index(train_index, train_dataset)
        except Exception as e:
            print(f"Error updating FAISS database at Epoch {epoch + 1}: {e}")
            
        # Evaluate the model
        evaluate(model, dev_mention_dict, dev_selected_keys, device)
            
    return model 

if __name__ == "__main__":
    import pickle

    # Load the training data
    with open("/home/zhiyong/Projects/llms_coref/project/corpus/ecb/filtered_mention_map_train.pkl", "rb") as f:
        mention_dict = pickle.load(f)
    selected_keys = list(mention_dict.keys())
    
    # Load the dev data
    with open("/home/zhiyong/Projects/llms_coref/project/corpus/ecb/filtered_mention_map_dev.pkl", "rb") as f:
        dev_mention_dict = pickle.load(f)
    dev_selected_keys = list(dev_mention_dict.keys())
    
    trained_model = train(mention_dict, selected_keys, 
                                           dev_mention_dict, dev_selected_keys,
                                           batch_size=2, epochs=10)
    # Save the trained model
    torch.save(trained_model.state_dict(), "/home/zhiyong/Projects/llms_coref/project/bi_encoder.pt")
    