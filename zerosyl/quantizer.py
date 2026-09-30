"""
This module contains an inference class for quantizing the syllable features to discrete IDs.
This involves two steps. First quantize with spherical K-means, then collapse certain codebook
entries to silences (see the original paper and research repo @ https://github.com/nicolvisser/ZeroSyl
for more info). But basically you need to identify whether each codebook entry represents a silence or
not and then have a mapping from original codebook id to the new contiguous and smaller vocab.
To train such a mapping you can follow one of two approaches. (1) Follow the agglomerative clustering
approach in ZeroSyl (2) OR use a VAD system and a small dataset to find out which clusters map to silences
most often.
"""


from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FeatureQuantizerConfig:
    n_clusters: int = 10_000
    d_model: int = 1024


class SphericalKMeans(nn.Module):
    def __init__(self, cfg: FeatureQuantizerConfig):
        super().__init__()
        self.cfg = cfg
        self.codebook = nn.Parameter(
            torch.zeros(cfg.n_clusters, cfg.d_model), requires_grad=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, t, d = features.shape
        flat = F.normalize(features.reshape(-1, d), p=2.0, dim=1)
        return (flat @ self.codebook.t()).argmax(dim=1).view(b, t)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> SphericalKMeans:
        ckpt = torch.load(path)
        model = cls(FeatureQuantizerConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        model.codebook.copy_(F.normalize(model.codebook, p=2.0, dim=1))
        return model.eval()


class FeatureQuantizer(nn.Module):
    KMEANS_FILENAME = "kmeans.pt"
    SILENCES_FILENAME = "silences.pt"
    MAPPING_FILENAME = "mapping.pt"

    def __init__(self, cfg: FeatureQuantizerConfig):
        super().__init__()
        self.cfg = cfg
        self.kmeans = SphericalKMeans(cfg)
        self.register_buffer("silences", torch.zeros(cfg.n_clusters, dtype=torch.bool))
        self.register_buffer("mapping", torch.arange(cfg.n_clusters, dtype=torch.long))

    @property
    def sil_token_id(self) -> int:
        return int((~self.silences).sum().item())

    @property
    def vocab_size(self) -> int:
        # Speech ids are 0 .. sil_token_id-1; silence occupies sil_token_id.
        return self.sil_token_id + 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        indices = self.kmeans(F.normalize(features, p=2.0, dim=-1))
        return self.mapping[indices]

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> FeatureQuantizer:
        directory = Path(directory)
        kmeans = SphericalKMeans.from_pretrained(directory / cls.KMEANS_FILENAME)
        silences = torch.load(directory / cls.SILENCES_FILENAME).bool()
        mapping = torch.load(directory / cls.MAPPING_FILENAME).long()

        model = cls(kmeans.cfg)
        model.kmeans = kmeans
        model.silences.copy_(silences)
        model.mapping.copy_(mapping)

        return model.eval()