# Chain-of-Ground: Improving GUI Grounding via Iterative Reasoning and Reference Feedback

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2512.01979-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.01979) [![PDF](https://img.shields.io/badge/PDF-Download-blue.svg?style=for-the-badge&logo=adobe&logoColor=white)](https://arxiv.org/pdf/2512.01979.pdf) [![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/chico-research/tpanel-ui)

</div>

## Overview
Chain-of-Ground is a training-free, multi-step framework for GUI grounding. This repo implements triple-layer and two-layer pipelines with iterative reasoning and reference feedback, supports Qwen3-VL and UI-TARS backends, and includes ready-to-run evaluation and visualization tools. We also introduce TPanel-UI, a 420-image dataset of industrial control panels with blur/mask distortions and JSON annotations for precise point grounding.

<div align="center">
  <img src="img/architecture.png" alt="Chain-of-Ground pipeline" width="90%" />
</div>


## Result

ScreenSpot-Pro

| Agent Model               | Development | Creative | CAD  | Scientific | Office | OS   | Avg  |
| ------------------------- | ----------- | -------- | ---- | ---------- | ------ | ---- | ---- |
| Triple-Step CoG*          | 69.2        | 65.7     | 57.9 | 69.3       | 80.9   | 71.9 | 68.4 |
| Dual-Step CoG*            | 66.4        | 63.1     | 59.5 | 70.9       | 80.9   | 62.3 | 66.7 |
| GTA-32B                   | 61.2        | 52.8     | 60.5 | 65.0       | 83.5   | 65.3 | 63.6 |
| Holo1.5-72B               | 63.5        | 62.5     | 51.3 | 64.2       | 79.6   | 59.7 | 63.3 |
| UI-TARS-1.5               | 63.9        | 50.4     | 58.2 | 69.3       | 79.6   | 51.0 | 61.6 |
| Qwen3-VL-32B-Instruct*    | 67.5        | 57.6     | 47.9 | 65.3       | 73.6   | 60.1 | 61.4 |
| Seed-1.5-VL               | 53.8        | 59.2     | 59.0 | 61.4       | 74.8   | 60.2 | 60.9 |
| Qwen2.5-VL-72B-Instruct   | 53.5        | 44.9     | 44.4 | 59.1       | 72.6   | 49.5 | 53.3 |
| GUI-Spontight-18B         | 52.6        | 44.8     | 51.6 | 53.5       | 70.4   | 45.9 | 52.8 |
| SE-GUI-7B                 | 44.5        | 37.2     | 42.1 | 54.7       | 70.4   | 38.8 | 47.3 |
| UI-7B-L4.57B              | 31.8        | 40.2     | 31.8 | 47.2       | 65.6   | 33.2 | 42.0 |
| JE-TARS-1                 | 27.4        | 34.0     | 32.2 | 52.4       | 68.7   | 26.0 | 39.5 |
| UI-TARS-72B               | 40.8        | 39.6     | 17.2 | 45.7       | 54.8   | 30.1 | 38.1 |
| GUI-GI-3B                 | 31.1        | 26.6     | 32.2 | 48.0       | 59.1   | 16.1 | 37.1 |
| Operator                  | 35.1        | 39.6     | 16.1 | 43.7       | 53.0   | 32.7 | 36.6 |
| JE-3B                     | 38.1        | 34.6     | 23.0 | 38.6       | 57.0   | 25.5 | 36.1 |
| SeDi-GU-146               | 35.1        | 29.0     | 31.8 | 43.3       | 50.9   | 25.0 | 35.9 |
| UGround-72B               | 31.1        | 35.8     | 13.8 | 50.0       | 51.3   | 25.5 | 34.5 |
| Claude 3 Sonnet           | -           | -        | -    | -          | -      | -    | 27.7 |
| Qwen2.5-7B-7B             | 29.1        | 24.9     | 13.8 | 31.1       | 45.7   | 22.4 | 27.6 |
| OS-Atlas-VB               | 17.7        | 17.9     | 10.3 | 24.4       | 27.4   | 16.8 | 18.9 |
| Aria-UI                   | 8.4         | 14.7     | 6.1  | 18.1       | 16.1   | 2.6  | 11.3 |
| CogAgent-18B              | 8.0         | 5.6      | 6.1  | 13.4       | 10.0   | 3.1  | 7.7  |
| SeeClick                  | 0.3         | 0.6      | 1.9  | 2.0        | 0.9    | 1.5  | 1.1  |
| GPT-4o                    | 0.7         | 0.6      | 1.5  | 1.2        | 0.9    | 0.0  | 0.8  |

ScreenSpot-Pro (Supplement)
| Model Combination                                  | Dev  | Creative | CAD  | Scientific | Office | OS   | Avg  |
| -------------------------------------------------- | ---- | -------- | ---- | ---------- | ------ | ---- | ---- |
| **Baselines (Single-Step)**                        |      |          |      |            |        |      |      |
| UI-TARS-1.5-7B                                     | 31.8 | 40.2     | 31.8 | 47.2       | 65.6   | 33.2 | 42.0 |
| Qwen3-VL-32B                                       | 67.5 | 57.6     | 47.9 | 65.3       | 73.6   | 60.1 | 61.4 |
| Qwen3-VL-235B                                      | 66.4 | 63.1     | 52.8 | 66.1       | 75.8   | 61.9 | 63.9 |
| Gemini-3-pro                                      | 80.3 | 75.8     | 51.6 | 74.8       | 85.7   | 68.9 | 72.4 |
| **Dual-Step CoG Variants**                         |      |          |      |            |        |      |      |
| Qwen3-VL-32B → Qwen3-VL-32B                        | 65.4 | 64.7     | 57.2 | 69.3       | 79.6   | 61.2 | 65.8 |
| Qwen3-VL-235B → Qwen3-VL-32B                       | 66.4 | 63.1     | 59.5 | 70.9       | 80.9   | 62.3 | 66.7 |
| Gemini-3-pro → Gemini-3-pro                       | 81.3 | 78.1     | 54.3 | 81.9       | 87.8   | 72 | 75.3 |
| **Triple-Step CoG Variants**                       |      |          |      |            |        |      |      |
| Qwen3-VL-235B → Qwen3-VL-32B → Qwen3-VL-235B       | 69.2 | 60.8     | 61.8 | 70.9       | 76.6   | 60.7 | 66.4 |
| Qwen3-VL-32B → Qwen3-VL-32B → Qwen3-VL-32B         | 69.8 | 66.3     | 58.5 | 68.5       | 78.3   | 66.3 | 67.5 |
| UI-TARS-1.5-7B → Qwen3-VL-235B → Qwen3-VL-32B      | 69.2 | 65.7     | 57.9 | 69.3       | 80.9   | 71.9 | 68.4 |
| Qwen3-VL-235B → Qwen3-VL-235B → Qwen3-VL-235B       | 69.6 | 62.4     | 56.9 | 69.7       | 77.0   | 59.2 | 65.6 |
| Qwen3-VL-32B → UI-TARS-1.5-7B → Qwen3-VL-32B        | 63.4 | 62.7     | 59.2 | 67.3       | 74.3   | 60.2 | 64.3 |
| Qwen3-VL-32B → Qwen3-VL-235B → Qwen3-VL-32B         | 70.9 | 64.7     | 52.6 | 70.9       | 78.3   | 66.9 | 66.7 |



## Repository Structure
- `models/ScreenSpot-pro` — screen grounding pipelines tailored for screenshot tasks
- `models/TPanel_UI` — triple-layer and two-layer UI grounding methods and baselines


## Installation

```
pip install pillow requests transformers torch tqdm matplotlib
```

## Setup
- Environment variables:
  - `OPENROUTER_API_KEY` for all models via OpenRouter
  - `DASHSCOPE_API_KEY` for qwen3-vl-32b-instruct

Example:

```
# macOS/Linux
export OPENROUTER_API_KEY=your_key_here
export DASHSCOPE_API_KEY=your_key_here

# Windows PowerShell
$Env:OPENROUTER_API_KEY="your_key_here"
$Env:DASHSCOPE_API_KEY="your_key_here"
```

## Quick Start

### TPanel-UI
- Dataset: https://huggingface.co/datasets/chico-research/tpanel-ui
- Download via CLI:
  ```
  hf download chico-research/tpanel-ui --repo-type dataset --local-dir ./tpanel-ui
  ```
- Directory layout:
  ```
  ./tpanel-ui/
    images/        # screenshots
    annotations/   # test samples (*.json)
  ```
- Batch evaluation:
  ```
  python cli/main.py \
    --mode triple \
    --model1 qwen/qwen3-vl-235b-a22b-instruct \
    --model2 qwen/qwen3-vl-235b-a22b-instruct \
    --model3 qwen/qwen3-vl-235b-a22b-instruct \
    --batch \
    --dataset_type tpanelui \
    --screens_dir ./tpanel-ui/images \
    --tests_dir ./tpanel-ui/annotations \
    --task all \
    --inst_style instruction \
    --gt_type positive \
    --output ./logs.json
  ```

### ScreenSpot-Pro
- Dataset:
  - Official: https://huggingface.co/datasets/likaixin/ScreenSpot-Pro
  - Repository: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding
- Download via CLI:
  ```
  hf download likaixin/ScreenSpot-Pro --repo-type dataset --local-dir ./screenspot-pro
  ```
- Directory layout:
  ```
  ./screenspot-pro/
    images/        # high-resolution screenshots
    annotations/   # annotations (bbox, instruction, ui_type, etc.)
  ```
- Batch evaluation:
  ```
  python cli/main.py \
    --mode triple \
    --model1 qwen/qwen3-vl-235b-a22b-instruct \
    --model2 qwen/qwen3-vl-235b-a22b-instruct \
    --model3 qwen/qwen3-vl-235b-a22b-instruct \
    --batch \
    --dataset_type sspro \
    --screens_dir ./screenspot-pro/images \
    --tests_dir ./screenspot-pro/annotations \
    --task all \
    --inst_style instruction \
    --gt_type positive \
    --output ./logs.json
  ```

Run pipelines via CLI (OpenRouter only):

```
# Dual-layer (UITars → Qwen) on a single image
python cli/main.py \
  --mode dual \
  --model1 bytedance/ui-tars-1.5-7b \
  --model2 qwen/qwen3-vl-235b-a22b-instruct \
  --instruction "Open Settings" \
  --image /path/to/screenshot.jpg

# Triple-layer Qwen on a single image
python cli/main.py \
  --mode triple \
  --model1 qwen/qwen3-vl-235b-a22b-instruct \
  --model2 qwen/qwen3-vl-235b-a22b-instruct \
  --model3 qwen/qwen3-vl-235b-a22b-instruct \
  --instruction "Open Settings" \
  --image /path/to/screenshot.jpg

# Batch evaluation (sspro or tpanelui)
python cli/main.py \
  --mode triple \
  --model1 qwen/qwen3-vl-235b-a22b-instruct \
  --model2 qwen/qwen3-vl-235b-a22b-instruct \
  --model3 qwen/qwen3-vl-235b-a22b-instruct \
  --batch \
  --dataset_type tpanelui \
  --screens_dir ./tpanel-ui/images \
  --tests_dir ./tpanel-ui/annotations \
  --task all \
  --inst_style instruction \
  --gt_type positive \
  --output ./logs.json
  
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
  "language": "en",
  "gt_type": "positive",
  "instruction_style": "instruction"
}
```

Notes:
- `bbox` is `[x1, y1, x2, y2]` in pixel coordinates; scripts normalize internally.
- For negative samples, `gt_type` is `negative` and `bbox` may be omitted.

## Evaluation (ScreenSpot-Pro)
The [eval_screenspot_pro.py](scripts/eval_screenspot_pro.py) script computes metrics and visualizations for ScreenSpot-Pro.

Example invocation (requires mapping in `model_factory.py`; otherwise use dynamic import):

```
python scripts/eval_screenspot_pro.py \
  --model_type qwen3vl_235b_triple \
  --screenspot_imgs /path/to/images \
  --screenspot_test /path/to/annotations \
  --task all \
  --inst_style instruction \
  --language en \
  --gt_type positive \
  --log_path ./logs.json
```




If your model file resides under `models/TPanel_UI` or `models/ScreenSpot-pro`, align import paths in [model_factory.py](scripts/model_factory.py) or use dynamic import as shown. Note: folders with characters like spaces, hyphens, or parentheses are not Python packages; dynamic import by file path avoids this.


## Pipeline
- Multi-stage architecture: initial detection → refinement → final validation.
- Normalized-to-pixel mapping: model outputs in `[0,1000]` are projected to image pixels and normalized to `[0,1]` for reporting.
- Robust parsing: structured `<tool_call>` JSON preferred; regex fallback for coordinates when formatting is degraded.
- Image scaling: `smart_resize` with factor 32 (Qwen pipelines) or 28 (UITars+Qwen hybrid) preserves fidelity while controlling resolution.
- Resilient inference: retries with exponential backoff for OpenRouter.

### Triple-Layer Qwen (TPanel_UI)
- Layer 1: initial detection with Qwen3-VL-235B.
- Layer 2: refinement given constraints (e.g., region of interest).
- Layer 3: final validation selecting the best candidate.

### Hybrid Triple-Layer (ScreenSpot-pro)
- UITars initial detection (OpenRouter) → Qwen refinement (OpenRouter) → Qwen finalization (OpenRouter).

## Dataset
- Images and JSON annotations with fields: `id`, `img_filename`, `img_size`, `bbox` (x1,y1,x2,y2), `platform`, `application`, `data_type/ui_type`, `instruction`, `language`, `gt_type`, `instruction_style`.
- Positive samples include a target `bbox`; negative samples are explicitly labeled with `gt_type=negative`.
- Dataset: https://huggingface.co/datasets/chico-research/tpanel-ui

## Experimental Setup
- Environment variables: `OPENROUTER_API_KEY` (OpenRouter).
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
