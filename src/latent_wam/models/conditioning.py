from __future__ import annotations

import hashlib
import re

import torch
import torch.nn as nn

from latent_wam.config import ExperimentConfig
from latent_wam.types import StudentInputs


def _stable_token_id(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return 2 + int.from_bytes(digest, "little") % (vocab_size - 2)


class HashTextEncoder(nn.Module):
    """Offline smoke-test tokenizer; not used for paper-quality training."""

    def __init__(self, width: int, vocab_size: int, max_tokens: int):
        super().__init__()
        self.output_dim = width
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size

    def forward(self, texts: list[str], device: torch.device):
        ids = torch.zeros(len(texts), self.max_tokens, dtype=torch.long, device=device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[\w']+|[^\w\s]", text.lower())[: self.max_tokens]
            if not tokens:
                tokens = ["<empty>"]
            values = [_stable_token_id(token, self.vocab_size) for token in tokens]
            ids[row, : len(values)] = torch.tensor(values, device=device)
            valid[row, : len(values)] = True
        half = self.output_dim // 2
        frequency = torch.exp(
            -torch.arange(half, device=device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0, device=device)) / max(half - 1, 1))
        )
        angles = ids.float().unsqueeze(-1) * frequency
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if features.shape[-1] < self.output_dim:
            features = torch.nn.functional.pad(features, (0, self.output_dim - features.shape[-1]))
        return features, valid


class FrozenT5TextEncoder(nn.Module):
    def __init__(self, model_name: str, max_tokens: int, local_files_only: bool):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.encoder = T5EncoderModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.output_dim = self.encoder.config.d_model
        self.max_tokens = max_tokens

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, texts: list[str], device: torch.device):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = self.encoder(**encoded).last_hidden_state
        return hidden, encoded["attention_mask"].bool()


def build_text_encoder(config: ExperimentConfig) -> nn.Module:
    width = config.model.predictor_embed_dim
    if config.model.text_backend == "t5":
        return FrozenT5TextEncoder(
            config.model.text_model,
            config.model.max_text_tokens,
            config.model.text_local_files_only,
        )
    if config.model.text_backend == "hash":
        return HashTextEncoder(
            width, config.model.hash_vocab_size, config.model.max_text_tokens
        )
    raise ValueError(f"Unknown text backend: {config.model.text_backend}")


class ConditioningEncoder(nn.Module):
    def __init__(self, config: ExperimentConfig, text_input_dim: int):
        super().__init__()
        width = config.model.predictor_embed_dim
        self.text_projection = nn.Linear(text_input_dim, width)
        self.proprio_projection = nn.Linear(config.action.max_proprio_dim, width)
        self.action_projection = nn.Linear(config.action.max_action_dim, width)
        self.embodiment_embedding = nn.Embedding(config.action.schema_buckets, width)
        self.schema_embedding = nn.Embedding(config.action.schema_buckets, width)
        self.type_embedding = nn.Embedding(5, width)
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        inputs: StudentInputs,
        text_features: torch.Tensor,
        text_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = inputs.proprio.device
        text = self.text_projection(text_features)
        proprio = self.proprio_projection(inputs.proprio) + self.type_embedding.weight[1]
        past_action = self.action_projection(inputs.past_actions) + self.type_embedding.weight[2]
        embodiment = self.embodiment_embedding(inputs.embodiment_ids).unsqueeze(1) + self.type_embedding.weight[3]
        schema = self.schema_embedding(inputs.schema_ids).unsqueeze(1) + self.type_embedding.weight[4]
        text = text + self.type_embedding.weight[0]
        memory = torch.cat([text, proprio, past_action, embodiment, schema], dim=1)
        valid = torch.cat(
            [
                text_valid,
                inputs.proprio_valid,
                inputs.past_action_valid,
                torch.ones(inputs.proprio.shape[0], 2, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        return self.norm(memory).to(text_features.dtype), valid


class GatedConditioningAdapter(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, query, memory, memory_valid):
        update, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_valid,
            need_weights=False,
        )
        return query + torch.tanh(self.gate).to(update.dtype) * update
