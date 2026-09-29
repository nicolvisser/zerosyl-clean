# ZeroSyl

Encoder for syllable-scale speech units. You already know the idea; the [research repo](https://github.com/nicolvisser/ZeroSyl) has the paper, training, and eval. This repo has better code especially if you want GPU-only code.

Open [`encode.ipynb`](encode.ipynb), run it, and copy whatever you need into your own repo. Sample audio is in [`data/`](data/).

## Setup

Python 3.14+ and an NVIDIA GPU. Peak finding uses CuPy for CUDA 13 (`cupy-cuda13x`). If your CUDA version is different, change that dependency in `pyproject.toml` before syncing.

```bash
uv sync
```

Checkpoints:

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

WavLM-Large also lives on the [UniLM release](https://github.com/microsoft/unilm/tree/master/wavlm). The ZeroSyl files are on this repo’s GitHub releases too.

## Encode

16 kHz mono, shaped `[B, 1, T]`. The notebook shows batching with lengths.

```python
import torch

from zerosyl.wavlm import WavLMEncoder
from zerosyl.zerosyl import ZeroSylPooler, ZeroSylPoolerConfig
from zerosyl.quantizer import FeatureQuantizer

wavlm = WavLMEncoder.from_unilm_checkpoint(
    "checkpoints/WavLM-Large.pt", num_layers=22
).cuda().eval()
pooler_cfg = ZeroSylPoolerConfig(output_layer=22, boundary_layer=13)
pooler = ZeroSylPooler(pooler_cfg).cuda().eval()
quantizer = FeatureQuantizer.from_pretrained("checkpoints/zerosyl-v040").cuda().eval()

wav = torch.randn(1, 1, 16000, device="cuda")

with torch.inference_mode():
    enc = wavlm(wav)
    pooled = pooler(enc.all_hidden_states, enc.seqlens)
    units = quantizer(pooled.segment_features)
    units = units.masked_fill(pooled.segment_pad_mask, -1)
```

`units` are codebook ids. `9115` is silence. `-1` is padding.

## License

ZeroSyl code is MIT. The WavLM encoder is Apache-2.0. See [NOTICE](NOTICE).

## Citation

```bibtex
@inproceedings{visser2026zerosyl,
  title     = {ZeroSyl: Simple Zero-Resource Syllable Tokenization for Spoken Language Modeling},
  author    = {Visser, Nicol and Malan, Simon and Slabbert, Danel and Kamper, Herman},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```
