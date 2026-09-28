# ZeroSyl (clean)

Minimal [ZeroSyl](https://github.com/nicolvisser/ZeroSyl) encoder: unsupervised syllable-scale units from WavLM-Large.

This is a small, dependency-light alternative to the research repo. Runtime needs **PyTorch** (plus NumPy). Peak finding uses **CuPy** by default; install it unless you pass `use_cupy=False` to the pooler (or `find_peaks`), which uses more RAM. No training, eval, or CLI. ZeroSyl code is MIT; the WavLM encoder is Apache-2.0 (see [NOTICE](NOTICE)).

See [`encode.ipynb`](encode.ipynb) for a full example, including batched audio.

## Setup

Python ≥ 3.14 and an NVIDIA GPU. The default CuPy wheel is CUDA 13 (`cupy-cuda13x`); swap it in `pyproject.toml` if you use another CUDA version. Skip CuPy only if you call the pooler with `use_cupy=False` — the PyTorch fallback uses more RAM.

```bash
uv sync
```

## Checkpoints

```
checkpoints/WavLM-Large.pt
checkpoints/zerosyl-v040/{kmeans.pt, mapping.pt, silences.pt}
```

Download WavLM-Large from the official [UniLM WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) release.
Download `zerosyl-v040/{kmeans.pt, mapping.pt, silences.pt}` from the releases on this GitHub repo.

Or you could run:

```bash
mkdir -p checkpoints/zerosyl-v040
curl -L -o checkpoints/WavLM-Large.pt \
  https://storage.googleapis.com/zerospeech-checkpoints/WavLM-Large.pt
curl -L -o checkpoints/zerosyl-v040/kmeans.pt \
  https://storage.googleapis.com/zerospeech-checkpoints/zerosyl-v040/kmeans.pt
curl -L -o checkpoints/zerosyl-v040/mapping.pt \
  https://storage.googleapis.com/zerospeech-checkpoints/zerosyl-v040/mapping.pt
curl -L -o checkpoints/zerosyl-v040/silences.pt \
  https://storage.googleapis.com/zerospeech-checkpoints/zerosyl-v040/silences.pt
```

The `zerosyl-v040` codebook has 10k clusters, and a remapping such that silences are collapsed to a single SIL id of `9115`.

## Encode

16 kHz mono waveform `[1, T]` (or a batched `[B, 1, T]` with lengths):

```python
import torch

from zerosyl.quantizer import FeatureQuantizer
from zerosyl.wavlm import WavLMEncoder
from zerosyl.zerosyl import ZeroSylPooler

wavlm = WavLMEncoder.from_unilm_checkpoint(
    "checkpoints/WavLM-Large.pt", num_layers=22
).cuda().eval()
pooler = ZeroSylPooler().cuda().eval()
quantizer = FeatureQuantizer.from_pretrained("checkpoints/zerosyl-v040").cuda().eval()

wav = torch.randn(1, 1, 16000, device="cuda")  # [B, 1, T], 16 kHz

with torch.inference_mode():
    enc = wavlm(wav)
    pooled = pooler(enc.all_hidden_states, enc.seqlens)
    units = quantizer(pooled.segment_features)
    units = units.masked_fill(pooled.segment_pad_mask, -1)
```

Padded positions are `-1`. Consecutive silence tokens are **not** merged, but you could if you need to. You can also quantize `pooled.framewise_features` for 50 Hz units (same id repeated for the duration of each segment); mask with `pooled.segment_indices < 0` so pad frames stay `-1`.

`ZeroSylPooler` returns a `ZeroSylPoolerOutput`. WavLM frames are 50 Hz (hop 320 at 16 kHz). `T` is frames, `S` is variable-rate segments (padded to the longest item in the batch), `D=1024`.

| Field | Shape | Meaning |
| --- | --- | --- |
| `segment_features` | `[B, S, D]` | Mean-pooled WavLM layer 22 features, one vector per syllable-like segment |
| `segment_pad_mask` | `[B, S]` | `True` on trailing pad segments used to align the batch |
| `segment_indices` | `[B, T]` | Frame → segment id (`-1` on padded frames). Cuts are peaks in layer-13 energy |
| `framewise_features` | `[B, T, D]` | Each pooled vector repeated so the stream is exactly 50 Hz (pad frames 0) |
| `unpooled_features` | `[B, T, D]` | Layer 22 frames before pooling (i.e. pure WavLM features) |
| `norms_smooth` | `[B, T]` | Smoothed, standardized L2 norms of layer 13 — the peak-detection signal |

`quantizer(segment_features)` maps each segment to a codebook id in `0 … 9114`, or **`9115` for silence**. After `masked_fill` with `segment_pad_mask`, pad positions are `-1`.

Sample clips: [`data/sample.flac`](data/sample.flac), [`data/sample_long.flac`](data/sample_long.flac).

## Attribution

`zerosyl/wavlm/` follows [Benjamin van Niekerk](https://github.com/bshall)'s WavLM encoder, derived from [HuggingFace Transformers](https://github.com/huggingface/transformers) WavLM (Apache-2.0). Official checkpoints are from [Microsoft UniLM](https://github.com/microsoft/unilm/tree/master/wavlm) (MIT). Details in [`NOTICE`](NOTICE).

## Acknowledgements

Thanks to [Benjamin van Niekerk](https://github.com/bshall) for the WavLM implementation this encoder is based on.

## Citation

```bibtex
@inproceedings{visser2026zerosyl,
  title     = {ZeroSyl: Simple Zero-Resource Syllable Tokenization for Spoken Language Modeling},
  author    = {Visser, Nicol and Malan, Simon and Slabbert, Danel and Kamper, Herman},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```
