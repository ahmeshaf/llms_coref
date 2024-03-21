import torch
import torch.nn as nn

from inspect import getfullargspec
from transformers import AutoModel, AutoTokenizer


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)


class CrossEncoder(nn.Module):
    def __init__(
        self,
        is_training=True,
        long=True,
        model_name="allenai/longformer-base-4096",
        linear_weights=None,
    ):
        super(CrossEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(["<m>", "</m>"], special_tokens=True)
            self.tokenizer.add_tokens(["<doc-s>", "</doc-s>"], special_tokens=True)
            self.tokenizer.add_tokens(["<g>"], special_tokens=True)
            self.tokenizer.add_tokens(["/wiki/"], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = AutoModel.from_pretrained(model_name)

        self.start_id = self.tokenizer.encode("<m>", add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode("</m>", add_special_tokens=False)[0]

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(
        self, input_ids, attention_mask, position_ids, global_attention_mask, arg1, arg2
    ):
        arg_names = set(getfullargspec(self.model).args)

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
        cls_vector = output.last_hidden_state[:, 0, :]

        arg1_vec = None
        if arg1 is not None:
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)

        return cls_vector, arg1_vec, arg2_vec

    def generate_model_output(
        self, input_ids, attention_mask, position_ids, global_attention_mask, arg1, arg2
    ):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(
            input_ids, attention_mask, position_ids, global_attention_mask, arg1, arg2
        )

        return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        global_attention_mask=None,
        arg1=None,
        arg2=None,
        lm_only=False,
        pre_lm_out=False,
    ):
        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            position_ids=position_ids,
            arg1=arg1,
            arg2=arg2,
        )
        if lm_only:
            return lm_output

        return self.linear(lm_output)


class CrossEncoderSumm(CrossEncoder):
    def __init__(
            self,
            long=True,
            tokenizer=None,
            lm_model=None,
            linear_weights=None,
    ):
        super(CrossEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.model = lm_model
        self.long = long

        self.start_id = self.tokenizer.encode("<m>", add_special_tokens=False)[0]
        self.end_id = self.tokenizer.encode("</m>", add_special_tokens=False)[0]

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)


def add_special_tokens(lm_model, tokenizer_, special_tokens):
    tokenizer_.add_tokens(special_tokens, special_tokens=True)
    lm_model.resize_token_embeddings(len(tokenizer_))


if __name__ == "__main__":
    # load a t5-small model and tokenizer
    model = AutoModel.from_pretrained('google-t5/t5-small')

    tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small')

    add_special_tokens(model, tokenizer, ['<m>', '</m>'])

    # load a CrossEncoderSumm
    crossencoder = CrossEncoderSumm(long=False, lm_model=model.encoder, tokenizer=tokenizer)
    crossencoder.to("cuda:0")
    print(crossencoder)

