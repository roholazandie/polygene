import torch
from torch import nn
from typing import List, Optional, Tuple, Union

from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.utils import logging
import torch.nn.functional as F

from gene_tokenizer import GeneTokenizer
from utils.data_collators import DataCollatorForLanguageModeling

logger = logging.get_logger(__name__)

import torch
import numpy as np


def calculate_gaussian_weights(labels, offset, num_bins, variance=1.0):
    """
    Calculate dynamic weights based on a Gaussian distribution centered at each label.
    :param labels: Tensor of shape (batch_size,) containing the labels.
    :param variance: The variance of the Gaussian distribution.
    :return: A tensor of shape (batch_size, num_classes) with Gaussian-based dynamic weights.
    """
    # if offset = 5, num_bins = 5, then class indices = [5, 6, 7, 8, 9]
    # Class indices [5, 6, 7, 8, 9]
    class_indices = torch.arange(offset, offset + num_bins).float()
    # labels = labels.unsqueeze(1)  # Reshape for broadcasting (batch_size, 1)

    # Calculate the Gaussian weights
    # Using broadcasting to compute the weight for each class for each sample
    weights = torch.exp(-0.5 * ((class_indices.view(-1, 1) - labels.float()) ** 2) / variance)
    weights /= (np.sqrt(2 * np.pi * variance))  # Normalize by the Gaussian coefficient

    return weights



class GeneBertForMaskedLM(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # weights = calculate_gaussian_weights(labels, offset=self.config.num_special_tokens, num_bins=5, variance=1.0)
            # masked_lm_loss = F.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1), weight=weights.view(-1))
            loss_fct = CrossEntropyLoss(label_smoothing=0.5)  # -100 index = padding token
            # loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


if __name__ == "__main__":
    num_bins = 5
    tokenizer = GeneTokenizer(num_bins)


    # Let's see how to use the model
    bert_config = BertConfig(vocab_size=len(tokenizer.vocab),
                             num_special_tokens=len(tokenizer.special_tokens_map),
                             max_position_embeddings=10,
                             # num_hidden_layers=1,
                             # num_attention_heads=1
                             )
    model = GeneBertForMaskedLM(config=bert_config)

    seq = [5, 5, 6, 6, 5, 7, 8, 7, 9, 9]
    inputs = [{"input_ids": torch.tensor(seq).unsqueeze(0), "attention_mask": torch.ones(1, len(seq)).long()}]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=0.50)

    inputs = data_collator(inputs)

    outputs = model(**inputs)
    loss = outputs.loss
    logits = outputs.logits
    print(loss, logits)

    # weights = calculate_gaussian_weights(torch.Tensor([8.0]), variance=1.0)