import os.path
import pickle
from collections import defaultdict
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import typer
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from .models import CrossEncoder
from ..helper import ensure_path, f1_score, recall, precision, accuracy
from .bi_encoder import BiEncoder
from .helper import (
    CombinedDataset,
    VectorDatabase,
    create_faiss_db,
    get_arg_attention_mask_wrapper,
    process_batch,
    tokenize_bi,
    tokenize_with_postive_condiates,
    tokenize_ce,
    forward_ab,
    generate_biencoder_embeddings,
    generate_biencoder_embeddings_withgrad,
)

app = typer.Typer()


@torch.no_grad
def evaluate(model, mention_dict, selected_keys, device, top_k=10, text_key="marked_doc"):
    # check the model is on the specified device
    model.to(device)
    tokenizer = model.tokenizer
    m_start_id = model.start_id
    m_end_id = model.end_id

    # tokenize the dev set
    tokenized_dev_dict = tokenize_bi(tokenizer, selected_keys, mention_dict, m_end_id)
    dev_dataset = Dataset.from_dict(tokenized_dev_dict).with_format(
        "torch"
    )  # list to torch tensor
    dev_dataset = dev_dataset.map(
        lambda batch: get_arg_attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True,
        batch_size=1,
    )  # contains embs
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
            mention_id = batch["unit_ids"][0]
            if mention_dict[mention_id]["tag_descriptor"] == "singleton":
                singleton_count += 1
                continue

            embeddings = process_batch(batch, model, device)
            # Retrieve the nearest neighbors
            neighbors_list = faiss_db.get_nearest_neighbors(embeddings, top_k + 1)
            # Remove the first neighbor, which is the mention itself
            neighbors_list = neighbors_list[0][1:]
            # Check if the correct mention is in the neighbors
            gold_label = batch["label"][0]
            neighbor_label_list = [neighbor["label"] for neighbor in neighbors_list]
            correct_mention_found = gold_label in neighbor_label_list
            if correct_mention_found:
                true_positives += 1
            total += 1

    print("true_positives:", true_positives)
    print("total:", total)
    recall = true_positives / total if total > 0 else 0
    print("recall:", recall)


def normalize_embeddings(embeddings):
    return embeddings / torch.norm(embeddings, dim=1).reshape((-1, 1))


def calculate_cluster_embeddings(train_clusters, embeddings, split_id2index):
    return {
        clus_id: torch.mean(
            embeddings[[split_id2index[m_id] for m_id in clus], :], dim=0
        )
        for clus_id, clus in train_clusters.items()
    }


def prepare_embeddings_for_loss_calculation(
    mention_map,
    train_clusters,
    m_id,
    bi_encoder,
    batch_size,
    device,
    cluster_embeddings,
    split_id2index,
):
    target_cluster_id = mention_map[m_id]["gold_cluster"]
    target_embedding = cluster_embeddings[target_cluster_id].reshape((-1, 1))

    other_clusters = [
        (cl_id, m_ids)
        for cl_id, m_ids in train_clusters.items()
        if cl_id != target_cluster_id
    ]
    other_clusters_embs = torch.vstack(
        [cluster_embeddings[cl_id] for cl_id, _ in other_clusters]
    )

    batch_splits_ids = [m_id] + [
        mid for mid in train_clusters[target_cluster_id] if mid != m_id
    ]
    batch_embeddings = generate_biencoder_embeddings_withgrad(
        mention_map, batch_splits_ids, bi_encoder, batch_size, device
    )
    batch_embeddings = normalize_embeddings(batch_embeddings)
    target_cluster_embedding = torch.mean(batch_embeddings, dim=0)

    return target_embedding, target_cluster_embedding, other_clusters_embs


def add_positive_candidates(mention_map):
    cluster2mentions = {}
    for m_id, mention in mention_map.items():
        cluster_id = mention["gold_cluster"]
        if cluster_id not in cluster2mentions:
            cluster2mentions[cluster_id] = []
        cluster2mentions[cluster_id].append(m_id)

    for mention_ids in cluster2mentions.values():
        for m_id in mention_ids:
            mention_map[m_id]["positive_candidates"] = mention_ids


def calculate_loss(target_embedding, target_cluster_embedding, other_clusters_embs):
    dot_product_with_other_clusters = torch.matmul(
        other_clusters_embs, target_embedding
    )
    hard_negs = torch.topk(dot_product_with_other_clusters, k=10).values
    curr_loss = -torch.dot(
        target_embedding.squeeze(), target_cluster_embedding.squeeze()
    ) + torch.log(torch.sum(torch.exp(hard_negs)))
    return curr_loss


def train_centroid_gpt(
    mention_map,
    train_selected_keys,
    train_clusters,
    dev_selected_keys,
    model_name="roberta-base-cased",
    long=False,
    text_key="marked_doc",
    learning_rate=0.1,
    batch_size=2,
    epochs=1,
    save_path="../models/bi_encoder/",
    device_id="cuda:0",
):
    device = torch.device(device_id)
    bi_encoder = BiEncoder(model_name=model_name, long=long, is_training=True)
    bi_encoder.to(device)
    bi_encoder.train()

    optimizer = torch.optim.Adam(bi_encoder.parameters(), lr=learning_rate)
    split_id2index = {m_id: i for i, m_id in enumerate(train_selected_keys)}

    for epoch in range(epochs):
        total_loss = 0.0
        embeddings = generate_biencoder_embeddings(
            mention_map, bi_encoder, train_selected_keys, batch_size, device
        )
        embeddings = normalize_embeddings(embeddings)

        cluster_embeddings = calculate_cluster_embeddings(
            train_clusters, embeddings, split_id2index
        )

        for i, m_id in tqdm(
            enumerate(train_selected_keys),
            desc=f"Epoch {epoch}: Training BiEncoder",
            total=len(train_selected_keys),
        ):
            optimizer.zero_grad()
            (
                target_embedding,
                target_cluster_embedding,
                other_clusters_embs,
            ) = prepare_embeddings_for_loss_calculation(
                mention_map,
                train_clusters,
                m_id,
                bi_encoder,
                batch_size,
                device,
                cluster_embeddings,
                split_id2index,
            )

            curr_loss = calculate_loss(
                target_embedding, target_cluster_embedding, other_clusters_embs
            )
            # print("curr loss:", curr_loss.item())
            curr_loss.backward()
            optimizer.step()

            total_loss += curr_loss.item()

        print(f"Average Loss Epoch {epoch}: {total_loss / len(train_selected_keys)}")

    return bi_encoder


def train_centroid(
    mention_map,
    train_selected_keys,
    train_clusters,
    dev_selected_keys,
    model_name="roberta-base-cased",
    long=False,
    text_key="marked_doc",
    learning_rate=0.1,
    batch_size=2,
    epochs=1,
    save_path="../models/bi_encoder/",
    device_id="cuda:0",
):
    device = torch.device(device_id)
    # Initialize the model and tokenizer
    bi_encoder = BiEncoder(model_name=model_name, long=long, is_training=True)
    bi_encoder.to(device)
    bi_encoder.train()

    optimizer = torch.optim.AdamW(
        [
            {"params": bi_encoder.model.parameters(), "lr": 0.0001},
            {"params": bi_encoder.linear.parameters(), "lr": 0.001},
        ]
    )
    split_id2index = {m_id: i for i, m_id in enumerate(train_selected_keys)}
    for i in range(epochs):
        evaluate(bi_encoder, mention_map, dev_selected_keys, device, text_key=text_key)
        total_loss = 0.0
        # stacked Torch.FloatTensor()
        embeddings = generate_biencoder_embeddings(
            mention_map, bi_encoder, train_selected_keys, 2, device
        )

        embeddings = embeddings / torch.norm(embeddings, dim=1).reshape((-1, 1))

        cluster_embeddings = {
            clus_id: torch.mean(
                embeddings[[split_id2index[m_id] for m_id in clus], :], dim=0
            )
            for clus_id, clus in train_clusters.items()
        }
        loss = torch.squeeze(torch.zeros(1, 1, requires_grad=True).to(device))
        for i, m_id in tqdm(
            enumerate(train_selected_keys),
            desc=f"epoch {i}: Training BiEncoder",
            total=len(train_selected_keys),
        ):
            optimizer.zero_grad()
            target_embedding = embeddings[split_id2index[m_id], :].reshape((-1, 1))
            target_cluster_id = mention_map[m_id]["gold_cluster"]
            # target_cluster_embedding = cluster_embeddings[target_cluster_id]
            other_clusters = [
                (cl_id, m_ids)
                for cl_id, m_ids in train_clusters.items()
                if cl_id != target_cluster_id
            ]

            other_clusters_embs = torch.vstack(
                [
                    cl_emb
                    for cl, cl_emb in cluster_embeddings.items()
                    if cl != target_cluster_id
                ]
            )
            dot_other_clusters = torch.squeeze(
                torch.matmul(other_clusters_embs, target_embedding)
            )
            indices = torch.topk(dot_other_clusters, k=10).indices
            other_clusters_mention_ids = [other_clusters[i][1][0] for i in indices]

            batch_splits_ids = [m_id] + [
                mid2 for mid2 in train_clusters[target_cluster_id] if mid2 != m_id
            ]

            batch_embeddings = generate_biencoder_embeddings_withgrad(
                mention_map, batch_splits_ids, bi_encoder, batch_size, device, text_key=text_key
            )

            batch_embeddings = batch_embeddings / torch.norm(
                batch_embeddings, dim=1
            ).reshape((-1, 1))

            other_clusters_embs = generate_biencoder_embeddings_withgrad(
                mention_map, other_clusters_mention_ids, bi_encoder, batch_size, device, text_key=text_key
            )
            other_clusters_embs = other_clusters_embs / torch.norm(
                other_clusters_embs, dim=1
            ).reshape((-1, 1))
            target_embedding = batch_embeddings[0]
            target_cluster_embedding = torch.mean(batch_embeddings, dim=0)

            dot_other_clusters = torch.squeeze(
                torch.matmul(other_clusters_embs, target_embedding)
            )

            hard_negs = torch.topk(dot_other_clusters, k=10).values

            curr_loss = -torch.dot(
                torch.squeeze(target_embedding), torch.squeeze(target_cluster_embedding)
            ) + torch.log(torch.sum(torch.exp(hard_negs)))
            curr_loss.backward()
            optimizer.step()
            total_loss += curr_loss.item()

            # if i % batch_size == batch_size - 1 or m_id == train_selected_keys[-1]:
            #
            #     loss.backward()
            #     optimizer.step()
            #     total_loss += loss.item()
            #     loss = torch.squeeze(torch.zeros(1, 1, requires_grad=True).to(device))

        print(total_loss / len(train_selected_keys))
    return bi_encoder


def train(
    mention_dict,
    train_selected_keys,
    dev_selected_keys,
    model_name="roberta-base-cased",
    long=False,
    text_key="marked_doc",
    learning_rate=0.00001,
    batch_size=2,
    epochs=1,
    save_path="../models/bi_encoder/",
):
    # Initialize the model and tokenizer
    bi_encoder = BiEncoder(model_name=model_name, long=long, is_training=True)

    tokenizer = bi_encoder.tokenizer
    m_start_id = bi_encoder.start_id
    m_end_id = bi_encoder.end_id

    # Tokenize anchors and positive candidates
    add_positive_candidates(mention_dict)
    tokenized_anchor_dict, tokenized_positive_dict = tokenize_with_postive_condiates(
        tokenizer, train_selected_keys, mention_dict, m_end_id, text_key=text_key
    )

    # Prepare datasets
    train_dataset = Dataset.from_dict(tokenized_anchor_dict).with_format("torch")
    train_dataset = train_dataset.map(
        lambda batch: get_arg_attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True,
        batch_size=batch_size,
    )

    train_dataset_positive_candidates = Dataset.from_dict(
        tokenized_positive_dict
    ).with_format("torch")
    train_dataset_positive_candidates = train_dataset_positive_candidates.map(
        lambda batch: get_arg_attention_mask_wrapper(batch, m_start_id, m_end_id),
        batched=True,
        batch_size=batch_size,
    )

    # DataLoaders
    combined_dataset = CombinedDataset(train_dataset, train_dataset_positive_candidates)
    train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bi_encoder.to(device)

    # FAISS database setup
    faiss_db = VectorDatabase()
    train_index, _ = create_faiss_db(train_dataset, bi_encoder, device=device)
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
    optimizer = torch.optim.Adam(bi_encoder.parameters(), lr=learning_rate)

    # start the evaluation
    evaluate(bi_encoder, mention_dict, dev_selected_keys, device)

    # Training loop
    bi_encoder.train()
    for epoch in range(epochs):
        # Initialize progress bar
        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}")

        for i, (anchor_batch, positive_batch) in enumerate(train_dataloader, start=1):
            # Process anchor, positive, and negative batches
            anchor_embeddings = process_batch(anchor_batch, bi_encoder, device)
            positive_embeddings = process_batch(positive_batch, bi_encoder, device)

            hard_negatives_batch = faiss_db.get_hard_negative(
                anchor_embeddings, anchor_batch["label"]
            )
            negative_embeddings = process_batch(
                hard_negatives_batch, bi_encoder, device
            )

            # Compute loss and update model
            loss = triplet_loss(
                anchor_embeddings, positive_embeddings, negative_embeddings, margin
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.update()
            pbar.set_postfix(
                {
                    "Iteration": f"{i}/{len(train_dataloader)}",
                    "Loss": f"{loss.item():.4f}",
                }
            )
            break

        pbar.close()
        # Save the model
        bi_encoder.save_model(f"{save_path}/checkpoint-{epoch}/")
        # torch.save(bi_encoder, f"{save_path}/checkpoint-{epoch}/")
        # torch.save(
        #     bi_encbi_encoderoder.state_dict(),
        #     f"{save_path}bi_encoder_{epoch}.pt",
        # )

        # Update the FAISS database
        try:
            train_index, _ = create_faiss_db(train_dataset, bi_encoder, device=device)
            faiss_db.set_index(train_index, train_dataset)
        except Exception as e:
            print(f"Error updating FAISS database at Epoch {epoch + 1}: {e}")

        # Evaluate the model
        evaluate(bi_encoder, mention_dict, dev_selected_keys, device)

    return bi_encoder


@app.command()
def train_biencoder(
    mention_map_path: str,
    model_name: str = "roberta-base-uncased",
    text_key="marked_sentence",
    long=False,
    batch_size: int = 2,
    epochs: int = 10,
    save_path: str = "model_save_path",
    learning_rate: float = 0.1,
):
    # Load the training and developing data
    with open(mention_map_path, "rb") as f:
        mention_map = pickle.load(f)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_split_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == "train"
    ]

    train_clusters = defaultdict(list)
    for m_id in train_split_ids:
        train_clusters[mention_map[m_id]["gold_cluster"]].append(m_id)

    dev_selected_ids = [
        m_id
        for m_id, m in mention_map.items()
        if m["men_type"] == "evt" and m["split"] == "dev"
    ]

    # Use arguments in the train function
    # trained_model = train(
    #     mention_map,
    #     train_split_ids,
    #     dev_selected_ids,
    #     model_name=model_name,
    #     long=long,
    #     text_key=text_key,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     learning_rate=learning_rate,
    #     save_path=save_path,
    # )
    trained_model = train_centroid(
        mention_map,
        train_split_ids,
        train_clusters,
        dev_selected_ids,
        model_name=model_name,
        long=long,
        text_key=text_key,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=save_path,
    )


def save_ce_model(scorer_folder, parallel_model):
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + "/linear.chkpt"
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + "/bert")
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + "/bert")


def predict_dpos(parallel_model, dev_ab, dev_ba, device, batch_size):
    n = dev_ab["input_ids"].shape[0]
    indices = list(range(n))
    # new_batch_size = batching(n, batch_size, len(device_ids))
    # batch_size = new_batch_size
    all_scores_ab = []
    all_scores_ba = []
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Predicting"):
            batch_indices = indices[i : i + batch_size]
            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)
            all_scores_ab.append(scores_ab.detach().cpu())
            all_scores_ba.append(scores_ba.detach().cpu())

    return torch.cat(all_scores_ab), torch.cat(all_scores_ba)


def train_ce(
    train_pairs,
    train_labels,
    dev_pairs,
    dev_labels,
    parallel_model,
    mention_map,
    output_folder,
    device,
    text_key="marked_sentence",
    max_sentence_len=512,
    batch_size=16,
    n_iters=50,
    lr_lm=0.00001,
    lr_class=0.001,
):
    bce_loss = torch.nn.BCELoss()
    # mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": parallel_model.module.model.parameters(), "lr": lr_lm},
            {"params": parallel_model.module.linear.parameters(), "lr": lr_class},
        ]
    )

    # debug
    # train_pairs = train_pairs[:10]
    # train_labels = train_labels[:10]
    # dev_pairs = dev_pairs[:10]
    # dev_labels = dev_labels[:10]

    tokenizer = parallel_model.module.tokenizer

    # prepare data
    train_ab, train_ba = tokenize_ce(
        tokenizer,
        train_pairs,
        mention_map,
        parallel_model.module.end_id,
        text_key=text_key,
        max_sentence_len=max_sentence_len,
    )
    dev_ab, dev_ba = tokenize_ce(
        tokenizer,
        dev_pairs,
        mention_map,
        parallel_model.module.end_id,
        text_key=text_key,
        max_sentence_len=max_sentence_len,
    )

    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)

    for n in range(n_iters):
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.0
        # new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        new_batch_size = batch_size
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc="Training"):
            optimizer.zero_grad()
            batch_indices = train_indices[i : i + new_batch_size]

            scores_ab = forward_ab(parallel_model, train_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, train_ba, device, batch_indices)

            batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)

            scores_mean = (scores_ab + scores_ba) / 2

            loss = bce_loss(scores_mean, batch_labels)

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()

        print(f"Iteration {n} Loss:", iteration_loss / len(train_pairs))
        # iteration accuracy
        dev_scores_ab, dev_scores_ba = predict_dpos(
            parallel_model, dev_ab, dev_ba, device, batch_size
        )
        dev_predictions = (dev_scores_ab + dev_scores_ba) / 2
        dev_predictions = dev_predictions > 0.5
        dev_predictions = torch.squeeze(dev_predictions)

        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev recall:", recall(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))
        if n % 2 == 0:
            scorer_folder = str(output_folder) + f"/scorer/chk_{n}"
            save_ce_model(scorer_folder, parallel_model)
            print(f"saved model at {n}")

    scorer_folder = str(output_folder) + "/scorer/"
    save_ce_model(scorer_folder, parallel_model)


@app.command()
def train_cross_encoder(
    dataset_folder,
    train_pairs_path,
    dev_pairs_path,
    output_folder: Path,
    model_name: str = "roberta-base",
    max_sentence_len: int = 512,
    is_long=False,
    device="cuda:0",
    text_key: str = "marked_sentence",
    batch_size: int = 20,
    epochs: int = 10,
):
    ensure_path(output_folder)
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", "rb"))
    evt_mention_map = {
        m_id: m for m_id, m in mention_map.items() if m["men_type"] == "evt"
    }

    train_pairs = list(pickle.load(open(train_pairs_path, "rb")))
    dev_pairs = list(pickle.load(open(dev_pairs_path, "rb")))

    train_labels = [
        int(mention_map[m1]["gold_cluster"] == mention_map[m2]["gold_cluster"])
        for m1, m2 in train_pairs
    ]
    dev_labels = [
        int(mention_map[m1]["gold_cluster"] == mention_map[m2]["gold_cluster"])
        for m1, m2 in dev_pairs
    ]

    device = torch.device(device)
    device_ids = list(range(1))

    scorer_module = CrossEncoder(
        is_training=True, model_name=model_name, long=is_long
    ).to(device)

    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)
    train_ce(
        train_pairs,
        train_labels,
        dev_pairs,
        dev_labels,
        parallel_model,
        evt_mention_map,
        output_folder,
        text_key=text_key,
        max_sentence_len=max_sentence_len,
        device=device,
        batch_size=batch_size,
        n_iters=epochs,
        lr_lm=0.000001,
        lr_class=0.0001,
    )


if __name__ == "__main__":
    app()
