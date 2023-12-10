"""
This module contains the implementation of a Bidirectional Encoder (BiEncoder) using the transformers library. 
"""

import torch
import torch.nn as nn

from inspect import getfullargspec
from transformers import AutoModel, AutoTokenizer


# helper function
def init_weights(m):
    """
    Initialize the weights of the Linear layers in the neural network.

    Parameters:
    ----------
    m : nn.Module

    Returns:
    -------
    None

    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization
        nn.init.uniform_(m.bias)  # Uniform initialization


class BiEncoder(nn.Module):
    def __init__(
        self,
        is_training=True,
        long=False,
        model_name="allenai/longformer-base-4096",
        linear_weights=None,
    ):
        super(BiEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(["<m>", "</m>"], special_tokens=True)
            self.tokenizer.add_tokens(["<doc-s>", "</doc-s>"], special_tokens=True)
            self.tokenizer.add_tokens(["<g>"], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()

        self.start_id = self.tokenizer.encode("<m>", add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode("</m>", add_special_tokens=False)[0]

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(
                self.hidden_size * 2, self.hidden_size
            ),  # 2 for CLS and one arg vector
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(
        self,
        input_ids,
        attention_mask,
        position_ids,
        global_attention_mask,  # NOTE this is not used in the function
        arg,
    ):
        arg_names = set(getfullargspec(self.model).args)
        # print(self.long)
        if self.long is True:
            output = self.model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
        else:
            output = self.model(input_ids, attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state
        cls_vector = output.pooler_output

        arg_vec = None
        if arg is not None:
            arg_vec = (last_hidden_states * arg.unsqueeze(-1)).sum(1)

        return cls_vector, arg_vec

    def generate_model_output(
        self,
        input_ids,
        attention_mask,
        position_ids,
        global_attention_mask,
        arg,
    ):
        cls_vector, arg_vec = self.generate_cls_arg_vectors(
            input_ids, attention_mask, position_ids, global_attention_mask, arg
        )
        return torch.cat([cls_vector, arg_vec], dim=1)

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        global_attention_mask=None,
        arg=None,
        lm_only=False,
        pre_lm_out=False,
    ):
        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            global_attention_mask=global_attention_mask,
            arg=arg,
        )
        if lm_only:
            return lm_output

        return self.linear(lm_output)

    def save_model(self, model_path):
        self.model.save_pretrained(model_path + "/bert")
        self.tokenizer.save_pretrained(model_path + "/bert")
        torch.save(self.linear.state_dict(), model_path + "/linear.pt")

