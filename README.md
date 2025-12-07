# Chain-of-Ground: Improving GUI Grounding via Iterative Reasoning and Reference Feedback

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2512.01979-b31b1b.svg)](https://arxiv.org/abs/2512.01979) [![PDF](https://img.shields.io/badge/PDF-Download-blue.svg)](https://arxiv.org/pdf/2512.01979.pdf) [![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/chico-research/tpanel-ui)

</div>

## Overview
Chain-of-Ground is a training-free, multi-step framework for GUI grounding. This repo implements triple-layer and two-layer pipelines with iterative reasoning and reference feedback, supports Qwen3-VL and UI-TARS backends, and includes ready-to-run evaluation and visualization tools. We also introduce TPanel-UI, a 420-image dataset of industrial control panels with blur/mask distortions and JSON annotations for precise point grounding.

<div align="center">
  <img src="architecture.png" alt="Chain-of-Ground pipeline" width="90%" />
</div>


## Repository Structure
- `models/ScreenSpot-pro` — screen grounding pipelines tailored for screenshot tasks
- `models/TPanel_UI` — triple-layer and two-layer UI grounding methods and baselines
- `eval_screenspot_pro.py` — evaluation and visualization utilities for dataset results
- `model_factory.py` — factory for loading models by type (adjust imports if needed)

## Installation

```
pip install pillow requests transformers torch tqdm matplotlib
```

## Setup
- Environment variables:
  - `DASHSCOPE_API_KEY` for Qwen3-VL via Dashscope
  - `OPENROUTER_API_KEY` for UI-TARS via OpenRouter

## Quick Start
Run a triple-layer Qwen method directly from file using dynamic import:

```
import importlib.util
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "qwen_triple", "models/TPanel_UI/Triple-layers/qwen3vl_235b_triple_mydata.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.Qwen3VL235BTripleMethod()
res = model.ground_only_positive("Open Settings", "path/to/screenshot.jpg")
print(res)
```

Hybrid triple-layer (UITars + Qwen) similarly:

```
spec = importlib.util.spec_from_file_location(
    "uitars_qwen_hybrid", "models/ScreenSpot-pro/Triple-layers/uitars1_5_7b_qwen3vl_235b_32b_RedCircle_BlueBox.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.UITarsQwen3VLHybridMethod()
res = model.ground_only_positive("Open File", "path/to/screenshot.jpg")
print(res)
```

## Dataset Format
Expected fields (example for a positive sample):

```
{
  "id": "sample_0001",
  "img_filename": "0001.png",
  "img_size": [1920, 1080],
  "bbox": [x1, y1, x2, y2],
  "data_source": "windows",
  "data_type": "text",
  "platform": "windows",
  "application": "explorer",
  "instruction": "Open Settings",
  "instruction_cn": "打开设置",
  "language": "en",
  "gt_type": "positive",
  "instruction_style": "instruction"
}
```

Notes:
- `bbox` is `[x1, y1, x2, y2]` in pixel coordinates; scripts normalize internally.
- For negative samples, `gt_type` is `negative` and `bbox` may be omitted.
- If Chinese (`language=cn`), current tooling supports `positive` with `instruction_style=instruction`.

## Evaluation
The `eval_screenspot_pro.py` script computes metrics and visualizations for a dataset.

Example invocation (adjust `--model_type` and paths as appropriate):

```
python eval_screenspot_pro.py \
  --model_type qwen3vl_235b_triple_mydata \
  --screenspot_imgs /path/to/images \
  --screenspot_test /path/to/test_jsons \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./logs.json
```

If your model file resides under `models/TPanel_UI` or `models/ScreenSpot-pro`, align import paths in `model_factory.py` or use dynamic import as shown.


## Pipeline
- Multi-stage architecture: initial detection → refinement → final validation.
- Normalized-to-pixel mapping: model outputs in `[0,1000]` are projected to image pixels and normalized to `[0,1]` for reporting.
- Robust parsing: structured `<tool_call>` JSON preferred; regex fallback for coordinates when formatting is degraded.
- Image scaling: `smart_resize` with factor 32 and pixel bounds preserves fidelity while controlling resolution.
- Resilient inference: retries with exponential backoff for Dashscope (Qwen3-VL) and OpenRouter (UI-TARS).

### Triple-Layer Qwen (TPanel_UI)
- Layer 1: initial detection with Qwen3-VL-235B.
- Layer 2: refinement given constraints (e.g., region of interest).
- Layer 3: final validation selecting the best candidate.

### Hybrid Triple-Layer (ScreenSpot-pro)
- UITars initial detection (OpenRouter) → Qwen refinement (Dashscope) → Qwen finalization (Dashscope).

## Dataset
- Images and JSON annotations with fields: `id`, `img_filename`, `img_size`, `bbox` (x1,y1,x2,y2), `platform`, `application`, `data_type/ui_type`, `instruction`, `instruction_cn`, `language`, `gt_type`, `instruction_style`.
- Positive samples include a target `bbox`; negative samples are explicitly labeled with `gt_type=negative`.
- Dataset: https://huggingface.co/datasets/chico-research/tpanel-ui

## Experimental Setup
- Environment variables: `DASHSCOPE_API_KEY` (Dashscope), `OPENROUTER_API_KEY` (OpenRouter).
- Recommended Python: 3.10+; packages: `Pillow`, `requests`, `transformers`, `torch`, `tqdm`, `matplotlib`.
- Seed: `torch.manual_seed(114514)` in `eval_screenspot_pro.py` for replicability.

## Metrics
- Overall accuracy (`action_acc`): fraction of correct predictions among all samples.
- Text/icon accuracy (`text_acc`, `icon_acc`): per-UI-type correctness.
- `wrong_format_num`: number of responses with invalid format.

## Results
We report overall accuracy and per-UI-type accuracy (text, icon). Example highlights:
- ScreenSpot-Pro: 68.4% accuracy (+4.8 points)
- TPanel-UI: +6.9 points over Qwen3‑VL‑235B

Fine-grained reports by platform, application, instruction style, and ground-truth type are produced by the evaluation script. Visualizations for the first N positive samples are saved to `./visualizations_{model_type}`.

## Limitations
- Reliance on external APIs may introduce latency and rate limits.
- Formatting deviations can still degrade parsing despite regex fallbacks.
- The Chinese setting is currently limited to positive samples with `instruction_style=instruction`.

## Citation

```
@misc{li2025chainofgroundimprovingguigrounding,
    title={Chain-of-Ground: Improving GUI Grounding via Iterative Reasoning and Reference Feedback}, 
    author={Aiden Yiliu Li and Bizhi Yu and Daoan Lei and Tianhe Ren and Shilong Liu},
    year={2025},
    eprint={2512.01979},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2512.01979}, 
}
```
