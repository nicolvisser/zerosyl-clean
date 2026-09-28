# Copyright 2021 The Fairseq Authors, Microsoft Research, and The HuggingFace Inc. team.
# Copyright 2026 Nicol Visser, Simon Malan, Danel Slabbert, Herman Kamper
#
# Derived from Benjamin van Niekerk's (@bshall) WavLM encoder, itself
# based on HuggingFace Transformers WavLM. Modified for ZeroSyl: layer-norm
# and hop/center padding, trimmed encoder, UniLM checkpoint loading.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    FeatureEncoder,
    FeatureProjection,
    PositionalConvEmbedding,
    TransformerLayer,
)
from .utils import map_from_unilm_checkpoint


@dataclass(frozen=True)
class WavLMEncoderConfig:
    conv_dims: tuple = (1, 512, 512, 512, 512, 512, 512, 512)
    conv_kernel_sizes: tuple = (10, 3, 3, 3, 3, 2, 2)
    conv_strides: tuple = (5, 2, 2, 2, 2, 2, 2)
    hidden_size: int = 1024
    layer_norm_eps: float = 1e-5
    feat_proj_dropout: float = 0.0
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    attention_dropout: float = 0.0
    num_buckets: int = 320
    max_bucket_distance: int = 800
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    hidden_dropout: float = 0.0
    activation_dropout: float = 0.0
    num_layers: int = 24

    @property
    def conv_hop_length(self) -> int:
        return math.prod(self.conv_strides)

    @property
    def conv_win_length(self) -> int:
        return 1 + sum(
            (x - 1) * math.prod(self.conv_strides[:i])
            for i, x in enumerate(self.conv_kernel_sizes)
        )


@dataclass(frozen=True)
class WavLMEncoderOutput:
    hidden_states: torch.Tensor  # [B, T, D]
    all_hidden_states: list[torch.Tensor]  # N x [B, T, D]
    seqlens: torch.Tensor  # [B,]


class WavLMEncoder(nn.Module):
    def __init__(self, cfg: WavLMEncoderConfig = WavLMEncoderConfig()):
        super().__init__()
        self.cfg = cfg

        self.feature_extractor = FeatureEncoder(
            cfg.conv_dims, cfg.conv_kernel_sizes, cfg.conv_strides
        )
        self.feature_projection = FeatureProjection(
            cfg.conv_dims[-1],
            cfg.hidden_size,
            cfg.layer_norm_eps,
            cfg.feat_proj_dropout,
        )

        self.pos_conv_embed = PositionalConvEmbedding(
            cfg.hidden_size,
            cfg.num_conv_pos_embeddings,
            cfg.num_conv_pos_embedding_groups,
        )
        self.dropout = nn.Dropout(cfg.hidden_dropout)

        self.layers = nn.ModuleList(
            TransformerLayer(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.num_attention_heads,
                cfg.attention_dropout,
                cfg.num_buckets,
                cfg.max_bucket_distance,
                cfg.hidden_dropout,
                cfg.activation_dropout,
                cfg.layer_norm_eps,
                has_relative_position_bias=(i == 0),
            )
            for i in range(cfg.num_layers)
        )

    @staticmethod
    def normalize(
        wav: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-utterance LayerNorm matching ZeroSyl ``F.layer_norm(wav, wav.shape)``.

        Official encode runs LayerNorm on each waveform before batch padding /
        right-pad. With ``lengths``, only the valid prefix of each item is
        normalized (batch padding zeros are kept at zero).
        """
        if lengths is None:
            return F.layer_norm(wav, wav.shape[1:])

        assert wav.ndim in (2, 3), f"Expected wav ndim 2 or 3, got {wav.ndim}"
        out = torch.zeros_like(wav)
        for i, length in enumerate(lengths.tolist()):
            length = int(length)
            if length <= 0:
                continue
            if wav.ndim == 3:
                sl = wav[i, :, :length]
                out[i, :, :length] = F.layer_norm(sl, sl.shape)
            else:
                sl = wav[i, :length]
                out[i, :length] = F.layer_norm(sl, sl.shape)
        return out

    def center_pad(self, wav):
        pad_amt = (self.cfg.conv_win_length - self.cfg.conv_hop_length) // 2
        return F.pad(wav, (pad_amt, pad_amt))

    def right_pad_to_hop(
        self,
        wav: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Right-pad waveforms so each valid length is a multiple of the CNN hop.

        Matches ZeroSyl ``encode_parallel``: ``pad = hop - (L % hop)``, which
        adds a full hop when ``L`` is already a multiple of ``hop``.
        """
        hop = self.cfg.conv_hop_length
        if lengths is None:
            pad = hop - (wav.size(-1) % hop)
            return F.pad(wav, (0, pad)), None

        pad_amts = hop - (lengths % hop)
        new_lengths = lengths + pad_amts
        max_len = int(new_lengths.max().item())
        if wav.size(-1) < max_len:
            wav = F.pad(wav, (0, max_len - wav.size(-1)))
        return wav, new_lengths

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
        normalize: bool = True,
        center_pad: bool = True,
        right_pad_to_hop: bool = True,
    ) -> WavLMEncoderOutput:
        if normalize:
            waveforms = self.normalize(waveforms, lengths)
        if right_pad_to_hop:
            waveforms, lengths = self.right_pad_to_hop(waveforms, lengths)
        if center_pad:
            waveforms = self.center_pad(waveforms)

        hidden_states = self.feature_extractor(waveforms).transpose(1, 2)
        hidden_states = self.feature_projection(hidden_states)

        bsz, max_len, _ = hidden_states.shape

        if lengths is None:
            feature_lengths = torch.full(
                size=(bsz,),
                fill_value=max_len,
                dtype=torch.long,
                device=waveforms.device,
            )
            key_padding_mask = None
        else:
            if center_pad:
                # lengths are pre-center-pad sample counts (after optional right-pad).
                feature_lengths = lengths // self.cfg.conv_hop_length
            else:
                # lengths are already-preprocessed sample counts (LN + right-pad +
                # center-pad), as produced by ``collate_fn_zerosyl``. Equivalent to
                # ``right_padded_len // hop``.
                feature_lengths = (
                    lengths - self.cfg.conv_win_length
                ) // self.cfg.conv_hop_length + 1

            positions = torch.arange(max_len, device=waveforms.device)
            key_padding_mask = positions.unsqueeze(0) >= feature_lengths.unsqueeze(1)

            hidden_states = hidden_states.masked_fill(key_padding_mask.unsqueeze(-1), 0)

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = self.dropout(hidden_states + position_embeddings)

        all_hidden_states = []
        position_bias = None

        for layer in self.layers:
            hidden_states, position_bias = layer(
                hidden_states,
                key_padding_mask=key_padding_mask,
                position_bias=position_bias,
            )
            all_hidden_states.append(hidden_states)

        return WavLMEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            seqlens=feature_lengths,
        )

    @classmethod
    def from_unilm_checkpoint(
        cls,
        checkpoint_path: str,
        cfg: WavLMEncoderConfig | None = None,
        num_layers: int | None = None,
    ) -> "WavLMEncoder":
        """Load a WavLM encoder from checkpoint, optionally trimmed to ``num_layers``."""
        wavlm_checkpoint = torch.load(checkpoint_path)
        if cfg is None:
            cfg = WavLMEncoderConfig()
        if num_layers is not None:
            cfg = dataclasses.replace(cfg, num_layers=num_layers)
        model = cls(cfg)
        model.load_state_dict(
            map_from_unilm_checkpoint(
                wavlm_checkpoint["model"], num_layers=cfg.num_layers
            )
        )
        return model.eval()
