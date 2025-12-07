import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import ModelOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ..language_model.llama import LlamaDecoderLayer, LlamaPreTrainedModel, LlamaRMSNorm


@dataclass
class CustomBaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    select_positions: Optional[torch.LongTensor] = None
    unselect_positions: Optional[torch.LongTensor] = None


@dataclass
class CustomSequenceClassifierOutputWithPast(ModelOutput):
    """Minimal selector output for inference."""

    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    new_input_embeds: Optional[torch.FloatTensor] = None
    new_attention_mask: Optional[torch.BoolTensor] = None
    select_tokens: Optional[torch.LongTensor] = None
    unselect_tokens: Optional[torch.LongTensor] = None
    select_layers: Optional[torch.LongTensor] = None
    selector_output_prob: Optional[torch.FloatTensor] = None


class Selector(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.mm_selector_num_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CustomBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        use_cache = False
        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=True,
                use_cache=use_cache,
                in_selector=True
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                # FIXME: maybe this is a way to prune self-attn without considering K-V cache
                # if decoder_layer.layer_idx not in self.config.mask_attn_layers:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                # FIXME: maybe this is a way to prune self-attn without considering K-V cache
                # if decoder_layer.layer_idx not in self.config.mask_attn_layers:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        select_positions = None
        unselect_positions = None

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return CustomBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            select_positions=select_positions,
            unselect_positions=unselect_positions
        )


class SelectorForClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.layer_select_tokens = nn.Parameter(torch.zeros(config.learnable_token_num, config.hidden_size))
        self.model = Selector(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    def selection_process(self, original_length, logits, inputs_embeds, attention_mask):
        select_tokens_prob = None

        image_tokens_logits = logits[0, 36: 36 + 576]
        layer_tokens_logits = logits[0, original_length: original_length + self.config.learnable_token_num]

        tokens_to_select = torch.cat([image_tokens_logits, layer_tokens_logits], dim=0)

        select_tokens_prob = torch.sigmoid(tokens_to_select)

        token_prune_ratio = self.config.token_prune_ratio if self.config.token_prune_ratio is not None else 0.0
        layer_prune_ratio = self.config.layer_prune_ratio if self.config.layer_prune_ratio is not None else 0.0

        select_image_tokens_num = int(576 * (1 - token_prune_ratio))
        select_layer_tokens_num = int(self.config.num_hidden_layers * (1 - layer_prune_ratio))

        image_tokens_prob = select_tokens_prob[:576, 0]
        layer_tokens_prob = select_tokens_prob[576:, 0]

        _, image_tokens_positions = torch.sort(image_tokens_prob, descending=True)
        _, layer_tokens_positions = torch.sort(layer_tokens_prob, descending=True)

        select_image_tokens_positions = image_tokens_positions[: select_image_tokens_num]
        unselect_image_tokens_positions = image_tokens_positions[select_image_tokens_num:]

        select_layer_tokens_positions = layer_tokens_positions[: select_layer_tokens_num]
        unselect_layer_tokens_positions = layer_tokens_positions[select_layer_tokens_num:]

        select_image_tokens_positions, _ = torch.sort(select_image_tokens_positions)
        unselect_image_tokens_positions, _ = torch.sort(unselect_image_tokens_positions)

        select_layer_tokens_positions, _ = torch.sort(select_layer_tokens_positions)
        unselect_layer_tokens_positions, _ = torch.sort(unselect_layer_tokens_positions)

        # Note this includes layer select tokens, and ==0 means that token is selected,
        # as we observe the initial state tends to be 0, and this may make the training process more stable.
        select_layers = select_layer_tokens_positions

        if self.config.mode != "mask":
            raise Exception("Only mask mode is supported in simplified selector.")

        select_image_tokens_positions = select_image_tokens_positions + 36
        unselect_image_tokens_positions = unselect_image_tokens_positions + 36

        all_ones = torch.ones([attention_mask.shape[0], original_length], device=attention_mask.device)
        all_ones.squeeze().index_put_([unselect_image_tokens_positions], torch.tensor(0.0, device=attention_mask.device))
        new_attention_mask = (all_ones == 1)

        return None, new_attention_mask, select_image_tokens_positions, unselect_image_tokens_positions, select_layers, select_tokens_prob


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        original_length: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        select_tokens: Optional[torch.LongTensor] = None,
        select_layers: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CustomSequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds_old = inputs_embeds

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        inputs_embeds_selected, attention_mask_selected, image_tokens_selected, image_tokens_unselected, layer_selected, selector_output_prob = self.selection_process(
            original_length, logits, inputs_embeds_old, attention_mask
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return output

        return CustomSequenceClassifierOutputWithPast(
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            new_input_embeds=inputs_embeds_selected,
            new_attention_mask=attention_mask_selected,
            select_tokens=image_tokens_selected,
            unselect_tokens=image_tokens_unselected,
            select_layers=layer_selected,
            selector_output_prob=selector_output_prob
        )
