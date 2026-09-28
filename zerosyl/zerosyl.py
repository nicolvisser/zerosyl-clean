from dataclasses import dataclass, field
from typing import override

import torch
from torch import nn

from .peaks import find_peaks
from .wavlm import WavLMEncoderConfig


@dataclass(frozen=True)
class ZeroSylPoolerConfig:
    wavlm: WavLMEncoderConfig = field(default_factory=WavLMEncoderConfig)
    output_layer: int = 22
    boundary_layer: int = 13
    win_size: int = 3
    prominence: float = 0.45


@dataclass(frozen=True)
class ZeroSylPoolerOutput:
    segment_features: torch.Tensor
    segment_pad_mask: torch.Tensor
    segment_indices: torch.Tensor
    unpooled_features: torch.Tensor
    norms_smooth: torch.Tensor

    @property
    def framewise_features(self) -> torch.Tensor:
        """Pooled segment vectors repeated at 50 Hz: ``[B, T, D]``.

        Frame ``t`` holds ``segment_features[b, segment_indices[b, t]]``.
        Padded frames are zeros.
        """
        idx = self.segment_indices
        d = self.segment_features.size(-1)
        gathered = self.segment_features.gather(
            1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, d)
        )
        return gathered.masked_fill(idx.unsqueeze(-1) < 0, 0)


class ZeroSylPooler(nn.Module):
    def __init__(self, cfg: ZeroSylPoolerConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ZeroSylPoolerConfig()
        self.cfg: ZeroSylPoolerConfig = cfg

    @override
    def forward(
        self,
        all_hidden_states: list[torch.Tensor],
        seqlens: torch.Tensor,
        use_cupy: bool = True,
    ):
        boundary_features = all_hidden_states[self.cfg.boundary_layer - 1]  # [B, T, D]
        unpooled_features = all_hidden_states[self.cfg.output_layer - 1]  # [B, T, D]

        boundaries_list, norms_smooth = extract_boundaries_from_norms(
            boundary_features,
            seqlens=seqlens,
            window_size=self.cfg.win_size,
            prominence=self.cfg.prominence,
            use_cupy=use_cupy,
        )

        segment_indices = build_segment_indices(
            boundaries_list=boundaries_list,
            seqlens=seqlens,
            max_seqlen=boundary_features.size(1),
            device=boundary_features.device,
        )
        segment_pad_mask = build_segment_pad_mask(segment_indices)
        segment_features = meanpool(unpooled_features, segment_indices)

        return ZeroSylPoolerOutput(
            segment_features=segment_features,
            segment_pad_mask=segment_pad_mask,
            segment_indices=segment_indices,
            unpooled_features=unpooled_features,
            norms_smooth=norms_smooth,
        )


def extract_boundaries_from_norms(
    boundary_features: torch.Tensor,
    seqlens: torch.Tensor,
    window_size: int,
    prominence: float,
    use_cupy: bool = True,
) -> tuple[list[torch.Tensor], torch.Tensor]:

    device = boundary_features.device
    boundary_features = boundary_features.to(torch.float32)

    norms = torch.linalg.vector_norm(boundary_features, dim=-1)

    # Standardize over valid frames only; padded frames would otherwise
    # shift the mean/std relative to single-item inference.
    # When doing unbatched/unpadded inference, this is equivalent to
    # norms = (norms - norms.mean()) / (norms.std() + 1e-9)
    positions = torch.arange(norms.size(1), device=device)
    valid = (positions.unsqueeze(0) < seqlens.unsqueeze(1)).to(norms.dtype)
    counts = seqlens.unsqueeze(1).to(norms.dtype)
    mean = (norms * valid).sum(dim=-1, keepdim=True) / counts
    var = ((norms - mean).pow(2) * valid).sum(dim=-1, keepdim=True) / counts
    norms = (norms - mean) / (var.sqrt() + 1e-9)

    norms_smooth = box_smooth_1d(norms, window_size, seqlens=seqlens)

    results = find_peaks(
        norms_smooth.detach(),
        prominence=prominence,
        seqlens=seqlens,
        use_cupy=use_cupy,
    )

    boundaries_list = []
    for (peaks_tensor, _), seqlen in zip(results, seqlens):
        upper_bounds = torch.cat([peaks_tensor, torch.tensor([seqlen], device=device)])
        boundaries_list.append(upper_bounds)

    return boundaries_list, norms_smooth


def box_smooth_1d(
    signal: torch.Tensor,
    window_size: int,
    seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    assert window_size % 2 == 1, f"got window_size {window_size}"
    assert window_size > 2, f"got window_size {window_size}"

    device = signal.device

    if seqlens is not None:
        # Replicate each row's last valid frame into its padded region so the
        # smoothing window near seqlen sees the same values as an unpadded run
        # (where replicate-padding extends the true last frame).
        positions = torch.arange(signal.size(1), device=device)
        gather_indices = torch.minimum(positions.unsqueeze(0), seqlens.unsqueeze(1) - 1)
        signal = signal.gather(1, gather_indices)

    kernel = (
        torch.ones(1, 1, window_size, device=device, dtype=signal.dtype) / window_size
    )
    pad_len = window_size // 2
    x_padded = torch.nn.functional.pad(
        signal.unsqueeze(1), (pad_len, pad_len), mode="replicate"
    )
    return torch.nn.functional.conv1d(x_padded, kernel).squeeze(1)


def normalize_boundaries(boundaries: torch.Tensor) -> torch.Tensor:
    """Prepare upper-bound cuts for torch.bucketize.

    Consecutive duplicates are removed. Non-positive bounds are dropped: a cut at
    0 creates a spurious one-frame first segment, and segment 0 already covers
    everything before the first positive upper bound.
    """
    boundaries = torch.unique_consecutive(boundaries)
    return boundaries[boundaries > 0]


def build_segment_indices(
    boundaries_list: list[torch.Tensor],
    seqlens: torch.Tensor,
    max_seqlen: int,
    device: torch.device,
) -> torch.Tensor:

    time_steps = torch.arange(max_seqlen, device=device)
    segment_indices = torch.stack(
        [
            torch.bucketize(time_steps, normalize_boundaries(boundaries), right=True)
            for boundaries in boundaries_list
        ]
    )

    valid_mask = time_steps.unsqueeze(0) < seqlens.unsqueeze(1)
    return torch.where(valid_mask, segment_indices, torch.tensor(-1, device=device))


def build_segment_pad_mask(segment_indices: torch.Tensor) -> torch.Tensor:
    num_segments = segment_indices.amax(dim=-1).clamp(min=-1) + 1
    max_segments = int(num_segments.amax().item())
    return torch.arange(
        max_segments, device=segment_indices.device
    ) >= num_segments.unsqueeze(1)


def meanpool(
    features: torch.Tensor,
    segment_indices: torch.Tensor,
) -> torch.Tensor:
    segment_pad_mask = build_segment_pad_mask(segment_indices)
    B, max_segments = segment_pad_mask.shape
    _, _, D = features.shape

    sums = torch.zeros(B, max_segments, D, device=features.device, dtype=features.dtype)
    counts = torch.zeros(
        B, max_segments, 1, device=features.device, dtype=features.dtype
    )

    valid_mask = segment_indices != -1
    safe_indices = segment_indices.clone()
    safe_indices[~valid_mask] = 0

    scatter_indices = safe_indices.unsqueeze(-1).expand(-1, -1, D)
    scatter_indices_counts = safe_indices.unsqueeze(-1)

    masked_features = features * valid_mask.unsqueeze(-1)
    masked_counts = valid_mask.unsqueeze(-1).to(features.dtype)

    sums.scatter_add_(1, scatter_indices, masked_features)
    counts.scatter_add_(1, scatter_indices_counts, masked_counts)

    pooled_features = sums / counts.clamp(min=1.0)
    return pooled_features * (~segment_pad_mask).unsqueeze(-1).to(features.dtype)
