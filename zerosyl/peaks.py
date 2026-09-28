import torch

# --- CUDA kernels ---

fill_left_min_src = r"""
extern "C" __global__ void fill_left_min(const float *signals, const int *peaks_row_indices, const int *peaks_col_indices, const int n_peaks,
                                         float *left_min, const int left_min_n_cols)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_peaks) return;

    int r = peaks_row_indices[i];
    int c = peaks_col_indices[i];

    float peak_height = signals[r * left_min_n_cols + c];
    float val = peak_height;

    for (int j = c - 1; j >= 0; j--) {
        float s = signals[r * left_min_n_cols + j];
        if (s > peak_height) break;
        if (s < val) val = s;
    }
    left_min[i] = val;
}
"""

fill_right_min_src = r"""
extern "C" __global__ void fill_right_min(const float *signals, const int *seqlens, const int *peaks_row_indices, const int *peaks_col_indices,
                                          const int n_peaks, float *right_min, const int right_min_n_cols)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_peaks) return;

    int r = peaks_row_indices[i];
    int c = peaks_col_indices[i];
    int seqlen = seqlens[r];

    float peak_height = signals[r * right_min_n_cols + c];
    float val = peak_height;

    for (int j = c + 1; j < seqlen; j++) {
        float s = signals[r * right_min_n_cols + j];
        if (s > peak_height) break;
        if (s < val) val = s;
    }
    right_min[i] = val;
}
"""

_fill_left_min_kernel = None
_fill_right_min_kernel = None


def _cupy_kernels():
    global _fill_left_min_kernel, _fill_right_min_kernel
    if _fill_left_min_kernel is None:
        import cupy as cp

        _fill_left_min_kernel = cp.RawKernel(fill_left_min_src, "fill_left_min")
        _fill_right_min_kernel = cp.RawKernel(fill_right_min_src, "fill_right_min")
    return _fill_left_min_kernel, _fill_right_min_kernel


# --- contour calculators --


def compute_contours_cupy(
    signal: torch.Tensor,
    unpadded_seqlens: torch.Tensor,
    peaks_row_indices: torch.Tensor,
    peaks_col_indices: torch.Tensor,
    peak_heights: torch.Tensor,
    n_threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    import cupy as cp

    assert n_threads % 32 == 0, f"got n_threads {n_threads}"
    seqlen = signal.shape[1]
    fill_left_min_kernel, fill_right_min_kernel = _cupy_kernels()

    left_min = peak_heights.clone()
    right_min = peak_heights.clone()

    signals_cp = cp.asarray(signal, dtype=cp.float32)
    unpadded_seqlens_cp = cp.asarray(unpadded_seqlens, dtype=cp.int32)
    peaks_row_indices_cp = cp.asarray(peaks_row_indices, dtype=cp.int32)
    peaks_col_indices_cp = cp.asarray(peaks_col_indices, dtype=cp.int32)
    left_min_cp = cp.asarray(left_min, dtype=cp.float32)
    right_min_cp = cp.asarray(right_min, dtype=cp.float32)

    n_peaks = peaks_row_indices_cp.size
    blocks = (n_peaks + n_threads - 1) // n_threads

    fill_left_min_kernel(
        (blocks,),
        (n_threads,),
        (
            signals_cp,
            peaks_row_indices_cp,
            peaks_col_indices_cp,
            n_peaks,
            left_min_cp,
            seqlen,
        ),
    )
    fill_right_min_kernel(
        (blocks,),
        (n_threads,),
        (
            signals_cp,
            unpadded_seqlens_cp,
            peaks_row_indices_cp,
            peaks_col_indices_cp,
            n_peaks,
            right_min_cp,
            seqlen,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    return (
        torch.as_tensor(left_min_cp, dtype=torch.float32, device=signal.device),
        torch.as_tensor(right_min_cp, dtype=torch.float32, device=signal.device),
    )


def compute_contours_torch(
    signal: torch.Tensor,
    unpadded_seqlens: torch.Tensor,
    peaks_row_indices: torch.Tensor,
    peaks_col_indices: torch.Tensor,
    peak_heights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    seqlen = signal.shape[1]
    n_peaks = peaks_row_indices.numel()

    idx = torch.arange(seqlen, device=signal.device).unsqueeze(0).expand(n_peaks, -1)
    peak_signals = signal[peaks_row_indices]
    higher_mask = peak_signals > peak_heights.unsqueeze(1)

    # --- left scan ---
    left_higher = higher_mask & (idx < peaks_col_indices.unsqueeze(1))
    left_bounds = (
        torch.where(left_higher, idx, torch.tensor(-1, device=signal.device)).max(
            dim=1
        )[0]
        + 1
    )
    valid_left = (idx >= left_bounds.unsqueeze(1)) & (
        idx <= peaks_col_indices.unsqueeze(1)
    )
    left_min = torch.where(
        valid_left, peak_signals, torch.tensor(float("inf"), device=signal.device)
    ).min(dim=1)[0]

    # --- right scan ---
    right_higher = higher_mask & (idx > peaks_col_indices.unsqueeze(1))
    right_bounds = (
        torch.where(right_higher, idx, torch.tensor(seqlen, device=signal.device)).min(
            dim=1
        )[0]
        - 1
    )
    right_bounds = torch.minimum(right_bounds, unpadded_seqlens[peaks_row_indices] - 1)
    valid_right = (idx >= peaks_col_indices.unsqueeze(1)) & (
        idx <= right_bounds.unsqueeze(1)
    )
    right_min = torch.where(
        valid_right, peak_signals, torch.tensor(float("inf"), device=signal.device)
    ).min(dim=1)[0]

    return left_min, right_min


# --- exports ---


def find_peaks(
    signal: torch.Tensor,
    prominence: float,
    seqlens: torch.Tensor | None = None,
    n_threads: int = 256,
    use_cupy: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor]]:

    assert signal.dtype is torch.float32, f"got dtype {signal.dtype}"

    bsz, seqlen = signal.shape

    if seqlens is None:
        unpadded_seqlens = torch.full(
            (bsz,), seqlen, dtype=torch.long, device=signal.device
        )
    else:
        unpadded_seqlens = seqlens

    # find (all) peaks
    is_peak = torch.zeros_like(signal, dtype=torch.bool)
    is_peak[..., 1:-1] = (signal[..., 1:-1] > signal[..., :-2]) & (
        signal[..., 1:-1] > signal[..., 2:]
    )

    # peaks cannot be on the last position or later
    seq_idx = torch.arange(seqlen, device=signal.device)
    no_peak_mask = seq_idx >= (unpadded_seqlens.unsqueeze(1) - 1)
    is_peak.masked_fill_(no_peak_mask, False)

    peaks_row_indices, peaks_col_indices = torch.where(is_peak)
    peak_heights = signal[peaks_row_indices, peaks_col_indices]

    # exit nicely if no peaks
    if peaks_row_indices.numel() == 0:
        empty_peaks = torch.tensor([], dtype=torch.long, device=signal.device)
        empty_proms = torch.tensor([], dtype=torch.float32, device=signal.device)
        return [(empty_peaks, empty_proms) for _ in range(bsz)]

    if use_cupy:
        left_min, right_min = compute_contours_cupy(
            signal,
            unpadded_seqlens,
            peaks_row_indices,
            peaks_col_indices,
            peak_heights,
            n_threads,
        )
    else:
        left_min, right_min = compute_contours_torch(
            signal, unpadded_seqlens, peaks_row_indices, peaks_col_indices, peak_heights
        )

    # prominence filtering
    contour_height = torch.maximum(left_min, right_min)
    prominences = peak_heights - contour_height

    valid = prominences > prominence
    peaks_row_indices = peaks_row_indices[valid]
    peaks_col_indices = peaks_col_indices[valid]
    prominences = prominences[valid]

    # format output
    output = []
    for b in range(bsz):
        mask = peaks_row_indices == b
        output.append(
            (
                peaks_col_indices[mask],
                prominences[mask],
            )
        )

    return output
