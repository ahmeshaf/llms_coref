import torch
from torch.utils.data import Dataset


class ECRSummarizationDataset(Dataset):
    '''
    Custom Dataset for ECR + Summarization tasks with mixed data
    '''
    def __init__(self, mention_map, ecr_data, ecr_labels, text_column, summarization_dataset):
        # Slicing summarization data for uniform task distribution
        num_ecr = len(ecr_labels)
        summarization_inputs = [f"Summarize the following article:\n\n{doc}" for doc in summarization_dataset['document'][:num_ecr]]
        summarization_labels = summarization_dataset['summary'][:num_ecr]

        ecr_inputs = [f"Event Coreference: {mention_map[m1]['mention_text']} in {mention_map[m1][text_column]}</s>{mention_map[m2]['mention_text']} in {mention_map[m2][text_column]}"
        for m1, m2 in ecr_data]

        self.inputs = ecr_inputs

        ecr_label_map = {
            1 : "Yes",
            0 : "No"
        }
        self.labels = [ecr_label_map[ecr_label] for ecr_label in ecr_labels]

        self.inputs.extend(summarization_inputs)
        self.labels.extend(summarization_labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class ECRDataset(Dataset):
    def __init__(self, mention_map, ecr_data, ecr_labels, text_column):

        ecr_inputs = [f"Event Coreference: {mention_map[m1]['mention_text']} in {mention_map[m1][text_column]}</s>{mention_map[m2]['mention_text']} in {mention_map[m2][text_column]}"
        for m1, m2 in ecr_data]

        self.inputs = ecr_inputs

        ecr_label_map = {
            1 : "Yes",
            0 : "No"
        }
        self.labels = [ecr_label_map[ecr_label] for ecr_label in ecr_labels]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class SummarizationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['document'])

    def __getitem__(self, idx):
        document = f"Summarize the following article:\n\n{self.data['document'][idx]}"
        summary = self.data['summary'][idx]
        id_ = self.data['id'][idx]


        return {'document': document, 'summary': summary, 'id': id_}