import math
import numpy as np
from functools import partial
from collections.abc import Callable
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.utils.parametrize as parametrize
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...activations import ACT2FN, gelu
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, check_model_inputs, maybe_autocast
from ...processing_utils import Unpack
from ...integrations import use_kernel_func_from_hub, use_kernelized_func
from ..xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from ...configuration_utils import PreTrainedConfig
from ..llama.modeling_llama import LlamaRotaryEmbedding, rotate_half, apply_rotary_pos_emb 
from ..xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaEmbeddings,
    XLMRobertaSelfAttention,
    XLMRobertaAttention,
    XLMRobertaIntermediate,
    XLMRobertaEncoder,
    XLMRobertaForMaskedLM,
    XLMRobertaForSequenceClassification,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
    eager_attention_forward,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPoolingAndNoAttention,
    BaseModelOutputWithPooling,
) 


logger = logging.get_logger(__name__)



# Tokenizer: `XLM-RoBERTa` tokenizer
# Model Architecture: Based on `jina-XLM-RoBERTa` model, with 5 LoRA adapters for 4 different tasks.

# 5 task-specific LoRA adapters are introduced to optimize embeddings for 4 tasks.
# retrieval.query
# retrieval.passage
# separation
# classification
# text-matching


# The task parameter is crucial and must be set according to the downstream task.
# The resulting embeddings will be optimized for that specific task.

# Note that the API does not first generate a generic meta embedding and then adapt it with an additional fine-tuned MLP.
# the task-specific LoRA adapter into every transformer layer (a total of 24 layers) and performs the encoding in one shot


# task, dimensions, and late_chunking.



# XLMRobertaLoRA
#   - XLMRobertaModel
#       - XLMRobertaPreTrainedModel
#           - PreTrainedModel



class JinaEmbeddingsV3Config(XLMRobertaConfig):
    model_type = "jina_embeddings_v3"

    def __init__(
        self,
        vocab_size=250002,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=8194,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="rotary",
        rotary_emb_base=20000.0,
        classifier_dropout=None,
        lora_adaptations=None,
        task_instructions=None,
        lora_rank=4,
        lora_dropout_p=0.0,
        lora_alpha=1,
        lora_main_params_trainable=False,
        load_trained_adapters=True,
        matryoshka_dimensions=None,
        emb_pooler=None,
        **kwargs,
    ):

        # 1. Set Defaults for mutable types (lists/dicts)
        if lora_adaptations is None:
            lora_adaptations = [
                "retrieval.query", "retrieval.passage", "separation",
                "classification", "text-matching"
            ]
        if matryoshka_dimensions is None:
            matryoshka_dimensions = [32, 64, 128, 256, 512, 768, 1024]

        # 2. Initialize the parent XLMRobertaConfig with standard BERT params
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            classifier_dropout=classifier_dropout,
            **kwargs,
        )

        # 3. Assign the Jina-specific parameters
        self.position_embedding_type = position_embedding_type
        self.rotary_emb_base = rotary_emb_base
        self.lora_adaptations = lora_adaptations
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout_p = lora_dropout_p
        self.lora_main_params_trainable = lora_main_params_trainable
        self.load_trained_adapters = load_trained_adapters
        self.matryoshka_dimensions = matryoshka_dimensions
        self.task_instructions = task_instructions
        self.emb_pooler = emb_pooler



class JinaEmbeddingsV3Embeddings(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        adapter_mask: torch.LongTensor | None = None,
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)
            if adapter_mask is not None:
                unique_tasks = torch.unique(adapter_mask)        
                for task_id in unique_tasks:
                    task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                    task_input_ids = input_ids[task_indices]
                    task_embeddings = self.word_embeddings(task_input_ids, task_id=task_id)
                    embeddings[task_indices] = task_embeddings
        else:
            embeddings = inputs_embeds

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids.expand(input_shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if adapter_mask is not None:
            unique_tasks = torch.unique(adapter_mask)
            for task_id in unique_tasks:
                task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                task_token_type_ids = token_type_ids[task_indices]
                task_token_type_embeddings = self.token_type_embeddings(token_type_ids, task_id=task_id)
                token_type_embeddings[task_token_type_ids] = task_token_type_embeddings

        embeddings = embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class JinaEmbeddingsV3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: JinaEmbeddingsV3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        # if self.rope_type != "default":
        #     rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: JinaEmbeddingsV3Config | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # base = config.rope_parameters["rope_theta"]
        base = config.rotary_emb_base
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@use_kernelized_func(apply_rotary_pos_emb)
class JinaEmbeddingsV3SelfAttention(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.attention_head_size * config.num_attention_heads)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adapter_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, 3, -1, self.attention_head_size)

        qkv = self.Wqkv(hidden_states).view(hidden_shape)
        query_states, key_states, value_states = qkv.unbind(dim=-3)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if adapter_mask is not None:
            unique_tasks = torch.unique(adapter_mask)
            for task_id in unique_tasks:
                task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                task_hidden_states = hidden_states[task_indices]
                task_qkv_states = self.Wqkv(task_hidden_states, task_id=task_id).view(hidden_shape)
                task_query_states, task_key_states, task_value_states = task_qkv_states.unbind(dim=-3)

                task_query_states = task_query_states.transpose(1, 2)
                task_key_states = task_key_states.transpose(1, 2)
                task_value_states = task_value_states.transpose(1, 2)

                query_states[task_indices] = task_query_states 
                key_states[task_indices] = task_key_states 
                value_states[task_indices] = task_value_states 

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights


class JinaEmbeddingsV3SelfOutput(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        adapter_mask: torch.Tensor | None = None,
    ):
        unique_tasks = torch.unique(adapter_mask)
        hidden_states = self.dense(hidden_states)
        for task_id in unique_tasks:
            task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
            task_hidden_states = hidden_states[task_indices]
            task_hidden_states = self.dense(task_hidden_states, task_id=task_id)
            hidden_states[task_indices] = task_hidden_states

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class JinaEmbeddingsV3Attention(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.attention_class = JinaEmbeddingsV3SelfAttention(config)
        self.output = JinaEmbeddingsV3SelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adapter_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        attention_output, attn_weights = self.attention_class(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            adapter_mask=adapter_mask,
            **kwargs,
        )
        attention_output = self.output(attention_output, hidden_states, adapter_mask=adapter_mask)
        return attention_output, attn_weights


class JinaEmbeddingsV3Intermediate(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor, adapter_mask: torch.Tensor | None = None) -> torch.Tensor:
        unique_tasks = torch.unique(adapter_mask)
        hidden_states = self.dense(hidden_states)
        for task_id in unique_tasks:
            task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
            task_hidden_states = hidden_states[task_indices]
            task_hidden_states = self.dense(task_hidden_states, task_id=task_id)
            hidden_states[task_indices] = task_hidden_states

        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class JinaEmbeddingsV3Output(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        adapter_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unique_tasks = torch.unique(adapter_mask)
        hidden_states = self.dense(hidden_states)
        for task_id in unique_tasks:
            task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
            task_hidden_states = hidden_states[task_indices]
            task_hidden_states = self.dense(task_hidden_states, task_id=task_id)
            hidden_states[task_indices] = task_hidden_states

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class JinaEmbeddingsV3Layer(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = JinaEmbeddingsV3Attention(config)
        self.intermediate = JinaEmbeddingsV3Intermediate(config)
        self.output = JinaEmbeddingsV3Output(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs]
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            position_embeddings,
            **kwargs,
        )

        layer_output = self.feed_forward_chunk(attention_output)
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class JinaEmbeddingsV3Encoder(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [JinaEmbeddingsV3Layer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adapter_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                position_embeddings,
                adapter_mask,
                **kwargs,
            )

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
        )


class JinaEmbeddingsV3PreTrainedModel(PreTrainedModel):
    pass



class JinaEmbeddingsV3Pooler(nn.Module):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        hidden_states: torch.Tensor,
        pool: bool = True,
        adapter_mask: torch.Tensor | None = None
    ):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        if adapter_mask is not None:
            unique_tasks = torch.unique(adapter_mask)
            for task_id in unique_tasks:
                task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                task_first_token_tensor = first_token_tensor[task_indices]
                task_pooled_output = self.dense(task_first_token_tensor, task_id=task_id)
                pooled_output[task_indices] = task_pooled_output

        pooled_output = self.activation(pooled_output)
        return pooled_output


def initialized_weights(
    shape: tuple[int], num_adaptations: int, init: str = "kaiming"
) -> torch.Tensor:
    weight_data = []
    for _ in range(num_adaptations):
        new_adaption = torch.zeros(shape)
        if init == "kaiming":
            nn.init.kaiming_uniform_(new_adaption, a=math.sqrt(5))
        elif init == "normal":
            nn.init.normal_(new_adaption)
        else:
            raise NotImplementedError
        weight_data.append(new_adaption)
    return torch.stack(weight_data, dim=0)


class LoRAParametrization(nn.Module):
    """
    This LoRA implementation was inspired by  https://github.com/cccntu/minLoRA
    The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
    LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: str = "linear",
        num_adaptations: int = 1,
        rank: int = 4,
        dropout_p: float = 0.0,
        alpha: float = 1,
    ):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        fan_in_fan_out = layer_type == "embedding"
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)

        if layer_type == "linear":
            self.lora_A = nn.Parameter(
                initialized_weights((rank, fan_in), num_adaptations, init="kaiming")
            )
            self.lora_B = nn.Parameter(torch.zeros((num_adaptations, fan_out, rank)))
        elif layer_type == "embedding":
            self.lora_A = nn.Parameter(torch.zeros((num_adaptations, fan_in, rank)))
            self.lora_B = nn.Parameter(
                initialized_weights(
                    (rank, fan_out), num_adaptations=num_adaptations, init="normal"
                )
            )
        else:
            raise NotImplementedError

        self.lora_alpha, self.rank = alpha, rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if dropout_p > 0 else lambda x: x
        self.register_buffer(
            "lora_dropout_mask",
            torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype),
            persistent=False,
        )

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def lora_forward(self, X, current_task):
        return (
            X
            + torch.matmul(
                *self.swap(
                    (
                        self.lora_B[current_task],
                        self.dropout_fn(self.lora_A[current_task]),
                    )
                )
            ).view(X.shape)
            * self.scaling
        )

    def forward(self, X):
        return X

    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        num_adaptations: int,
        rank: int,
        dropout_p: float,
        alpha: float,
    ):
        assert isinstance(layer, nn.Linear)
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            num_adaptations=num_adaptations,
            layer_type="linear",
            rank=rank,
            dropout_p=dropout_p,
            alpha=alpha,
        )

    @classmethod
    def from_embedding(
        cls,
        layer: nn.Module,
        num_adaptations: int,
        rank: int,
        dropout_p: float,
        alpha: float,
    ):
        assert isinstance(layer, nn.Embedding)
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in,
            fan_out,
            num_adaptations=num_adaptations,
            layer_type="embedding",
            rank=rank,
            dropout_p=dropout_p,
            alpha=alpha,
        )

    @classmethod
    def add_to_layer(
        cls,
        layer: nn.Module,
        num_adaptations: int,
        rank: int,
        dropout_p: float,
        alpha: float,
    ):
        """
        Registering LoRA adapters to all embedding and linear layers.
        Additionally, we implement a custom forward function for LoRA parametrization.
        This function modifies the layer's forward pass to optionally use task-specific
        parameters. When a `task_id` is provided, it employs a LoRA parametrization
        to modify the original weights according to the specific task. This allows
        the layer to adapt dynamically to different tasks at runtime. If no `task_id`
        is specified, the layer uses its original weights.
        """
        if isinstance(layer, nn.Linear):
            parametrize.register_parametrization(
                layer,
                "weight",
                cls.from_linear(
                    layer,
                    num_adaptations=num_adaptations,
                    rank=rank,
                    dropout_p=dropout_p,
                    alpha=alpha,
                ),
            )

            def new_forward(self, input, task_id=None, residual=False):
                if task_id is not None:
                    weights = self.parametrizations.weight[0].lora_forward(
                        self.weight, current_task=task_id
                    )
                else:
                    weights = self.weight

                out = F.linear(input, weights, self.bias)

                if residual:
                    return out, input
                return out

            layer.forward = new_forward.__get__(layer, layer.__class__)

        elif isinstance(layer, nn.Embedding):
            parametrize.register_parametrization(
                layer,
                "weight",
                cls.from_embedding(
                    layer,
                    num_adaptations=num_adaptations,
                    rank=rank,
                    dropout_p=dropout_p,
                    alpha=alpha,
                ),
            )

            def new_forward(self, input, task_id=None):
                if task_id is not None:
                    weights = self.parametrizations.weight[0].lora_forward(
                        self.weight, current_task=task_id
                    )
                else:
                    weights = self.weight

                out = F.embedding(
                    input,
                    weights,
                    self.padding_idx,
                    # self.max_norm,
                    # self.norm_type,
                    # self.scale_grad_by_freq,
                    # self.sparse,
                )

                return out

            layer.forward = new_forward.__get__(layer, layer.__class__)


@auto_docstring
class JinaEmbeddingsV3Model(nn.Module):
    _no_split_modules = ["JinaEmbeddingsV3Embeddings", "JinaEmbeddingsV3Layer"]

    def __init__(self, config: JinaEmbeddingsV3Config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__()
        self.config = config

        self.embeddings = JinaEmbeddingsV3Embeddings(config)
        self.encoder = JinaEmbeddingsV3Encoder(config)
        self.pooler = JinaEmbeddingsV3Pooler(config) if add_pooling_layer else None
        self.rotary_emb = JinaEmbeddingsV3RotaryEmbedding(config)

        self._setup_lora_config()
        self._register_lora()

        # Initialize weights and apply final processing
        # self.post_init()


    def _setup_lora_config(self):
        self._lora_adaptations = self.config.lora_adaptations
        if (
            not isinstance(self._lora_adaptations, list)
            or len(self._lora_adaptations) < 1
        ):
            raise ValueError(
                f"`lora_adaptations` must be a list and contain at least one element"
            )
        self._task_instructions = self.config.task_instructions
        if (
            not isinstance(self._task_instructions, dict)
            or len(self._task_instructions) != len(self._lora_adaptations)
            or not all(
                [v in self._lora_adaptations for v in self._task_instructions.keys()]
            )
        ):
            raise ValueError(
                f"`task_instructions` must be a dict and contain the same number of elements "
                f"as `lora_adaptations` with all keys in `task_instructions` present in `lora_adaptations`."
            )
        self._adaptation_map = {
            name: idx for idx, name in enumerate(self._lora_adaptations)
        }


    def _register_lora(self):
        self.apply(
            partial(
                LoRAParametrization.add_to_layer,
                num_adaptations=len(self._lora_adaptations),
                rank=self.config.lora_rank,
                dropout_p=self.config.lora_dropout_p,
                alpha=self.config.lora_alpha,
            )
        )

        # Handle freezing/unfreezing main parameters
        # if not self.config.lora_main_params_trainable:
        #     for name, param in self.named_parameters():
        #         if "lora" not in name:
        #             param.requires_grad_(False)


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        adapter_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) ->  BaseModelOutputWithPooling | tuple:

        if (input_ids is None) and (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape[0], input_shape[1]

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = (input_ids != self.config.pad_token_id).long()
            else:
                # Cannot infer padding from embeddings alone, defaulting to all ones
                attention_mask = torch.ones(input_shape, device=device, dtype=torch.long)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            adapter_mask=adapter_mask,
        )

        position_embeddings = self.rotary_emb(embedding_output, position_ids)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            position_embeddings=position_embeddings,
            adapter_mask=adapter_mask,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

    # @torch.inference_mode()
    # def encode(
    #     self,
    #     sentences: str | list[str],
    #     batch_size: int = 32,
    #     show_progress_bar: bool | None = None,
    #     output_value: str = "sentence_embedding",
    #     convert_to_numpy: bool = True,
    #     convert_to_tensor: bool = False,
    #     device: torch.device | None = None,
    #     normalize_embeddings: bool = True,
    #     truncate_dim: int | None = None,
    #     adapter_mask: torch.Tensor | None = None,
    #     task: str | None = None,
    #     **tokenizer_kwargs,
    # ) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
    #
    #     if task and task not in self._lora_adaptations:
    #         raise ValueError(
    #             f"Unsupported task '{task}'. "
    #             f"Supported tasks are: {', '.join(self.config.lora_adaptations)}."
    #             f"Alternatively, don't pass the `task` argument to disable LoRA."
    #         )
    #     adapter_mask = None
    #     if task:
    #         task_id = self._adaptation_map[task]
    #         num_examples = 1 if isinstance(sentences, str) else len(sentences)
    #         adapter_mask = torch.full(
    #             (num_examples,), task_id, dtype=torch.int32, device=self.device
    #         )
    #         if isinstance(sentences, str):
    #             sentences = self._task_instructions[task] + sentences
    #         else:
    #             sentences = [
    #                 self._task_instructions[task] + sentence for sentence in sentences
    #             ]
    #
    #
    #
    #     return self.roberta.encode(
    #         sentences, *args, adapter_mask=adapter_mask, **kwargs
    #     )
