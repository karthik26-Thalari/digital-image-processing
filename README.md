<div align="center">

# 🌫️ VNDHR + U-Net — Image Dehazing (Day & Night)

**Two-stage dehazing: classical variational optimization + deep learning refinement**

[![Demo](https://img.shields.io/badge/🤗_Live_Demo-HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/mayee1802/Image-Dehazing)
[![Paper](https://img.shields.io/badge/📄_Paper-IEEE_TITS_2025-00629B?style=for-the-badge)](https://ieeexplore.ieee.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-Academic_Research-green?style=for-the-badge)](#-license)

</div>

---

## 📌 Overview

A **research implementation** of a two-stage image dehazing pipeline supporting both **nighttime** and **daytime/morning** haze, based on:

> **"VNDHR: Variational Single Nighttime Image Dehazing for Enhancing Visibility in Intelligent Transportation Systems via Hybrid Regularization"**
> — *IEEE Transactions on Intelligent Transportation Systems (TITS), 2025*

🤗 **Try it live:** [huggingface.co/spaces/mayee1802/Image-Dehazing](https://huggingface.co/spaces/mayee1802/Image-Dehazing)

### How it works

```
Upload Any Hazy Image
        │
        ▼
┌───────────────────────┐
│  Auto Scene Detection  │  Brightness analysis (HSV V-channel)
└────────┬──────────────┘
         │
    ┌────┴────┐
    ▼         ▼
🌙 Night    ☀️ Day/Morning
    │              │
    ▼              ▼
┌────────┐   ┌──────────┐
│ VNDHR  │   │   DCP    │  Dark Channel Prior
│Variati.│   │Classical │  Atmospheric light
│  PCG   │   │Dehazing  │  Transmission map
└───┬────┘   └────┬─────┘
    └──────┬───────┘
           ▼
  ┌─────────────────┐
  │  Stage 2 — UNet │  Deep learning refinement
  │  5-Level Encoder│  BatchNorm + skip connections
  │  Perceptual Loss│  MSE + L1 + VGG perceptual
  └────────┬────────┘
           ▼
   Clear, Defogged Image ✅
```

---

## 📊 Results

| Method | PSNR (dB) | SSIM | Improvement |
|---|---|---|---|
| Hazy Input (baseline) | — | — | — |
| VNDHR only | 19.79 | 0.8470 | Classical baseline |
| **VNDHR + U-Net (ours)** | **26.01** | **0.9011** | **+6.22 dB / +0.054 SSIM** |
| Paper target | 28.0 | 0.8000 | ← SSIM exceeded ✅ |

**Key takeaways:**
- 🏆 **SSIM 0.9011 exceeds the paper target by 12.6%** — excellent structural preservation
- 📈 **+6.22 dB PSNR** gain over VNDHR-only baseline, validating the two-stage design
- 🌐 Competitive with state-of-the-art: O-Haze (23–25 dB), I-Haze (24–27 dB)
- 🧪 Evaluated on **1,000 test images** at 256×256 resolution

---

## 🔬 Pipeline Details

### Stage 1A — VNDHR Variational Model (Nighttime)

Decomposes a hazy image **S** into **Illumination (I)** and **Reflectance (R)** components using the hybrid energy functional (paper §III-B):

$$\min_{I,R} \|S - I \cdot R\|^2 + \lambda_1\|\nabla I\|_p + \lambda_2 W_r\|\nabla R\|_2^2 + \lambda_3\|\nabla R\|_1$$

| Parameter | Value | Role |
|---|---|---|
| λ₁ | 0.002 | ℓp norm weight — illumination smoothness |
| λ₂ | 0.0001 | Weighted ℓ2 — reflectance regularization |
| λ₃ | 0.001 | ℓ1 TV norm — noise suppression |
| p | 0.65 | Fractional-order norm exponent |
| max_iter | 10 | PCG optimization iterations |

Each sub-problem solved via **Preconditioned Conjugate Gradient (PCG)** on the V-channel of the HSV color space.

---

### Stage 1B — Dark Channel Prior (Daytime / Morning)

For daytime haze, the **Dark Channel Prior (DCP)** is applied:

1. Compute the **dark channel** — minimum intensity across color channels in a local patch
2. Estimate **atmospheric light** from the top 0.1% brightest pixels in the dark channel
3. Compute **transmission map** and smooth with Gaussian filter
4. Recover the **scene radiance** via the haze imaging model: `J = (I - A) / t + A`

This handles fog, smog, and morning haze conditions where VNDHR is not optimized.

---

### Stage 2 — U-Net Refinement (3-Stage Curriculum Training)

**Architecture:** 5-level U-Net · ~18M parameters · BatchNorm · skip connections · AdamW · AMP · 4-way TTA at inference

| Stage | Epochs | Learning Rate | Loss Weights | Augmentation |
|---|---|---|---|---|
| 1 — Perceptual intro | 0 – 40 | 3e-5 | MSE + L1 + VGG (gradual ramp) | Light |
| 2 — Fine-tuning | 41 – 70 | 5e-6 | MSE 60% + L1 40% | Medium |
| 3 — Aggressive push | 71 – 100 | 5e-5 | MSE 60% + L1 40% | Heavy |

---

## 🗂️ Project Structure

```
digital-image-processing-main/
│
├── codes/
│   ├── enhanced.ipynb              # ⭐ Main: VNDHR preprocessing + UNet training + eval
│   ├── research paper code.ipynb   # Experimental / earlier version
│   └── raw/                        # Draft notebooks
│
├── dataset/
│   ├── train/
│   │   ├── input/                  # Hazy training images
│   │   └── target/                 # Ground-truth clear images
│   └── test/
│       ├── input/                  # Hazy test images
│       └── target/                 # Ground-truth clear images
│
├── outputs/
│   ├── COMPARISON_4_PANEL/         # Side-by-side: hazy | VNDHR | UNet | GT
│   ├── enhanced results/
│   │   ├── comparison_images/
│   │   ├── dehazed_images/         # Final dehazed outputs
│   │   ├── metrics_results.csv     # Per-image PSNR & SSIM
│   │   └── summary_report.txt      # Full evaluation summary
│   ├── VNDHR_METRICS/
│   │   ├── vndhr_detailed_metrics.csv
│   │   └── vndhr_metrics_report.txt
│   └── train_vndhr/                # VNDHR-preprocessed training images
│
├── trained unet model/
│   └── unet_vndhr_aggressive.pth   # ⭐ Best model checkpoint
│
├── paper/
│   └── VNDHR_...IEEE_TITS.pdf      # Reference paper (IEEE TITS 2025)
│
└── README.md
```

---

## ⚙️ Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.8+ |
| CUDA GPU | 6 GB+ VRAM (recommended) |
| Jupyter | Notebook or JupyterLab |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/karthik26-Thalari/digital-image-processing.git
cd digital-image-processing
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

**With CUDA (GPU — recommended):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow numpy scipy scikit-image tqdm jupyter
```

**CPU only:**
```bash
pip install torch torchvision torchaudio
pip install opencv-python pillow numpy scipy scikit-image tqdm jupyter
```

### 4. Organize your dataset

```
dataset/
├── train/
│   ├── input/     ← hazy training images
│   └── target/    ← clear ground-truth images
└── test/
    ├── input/     ← hazy test images
    └── target/    ← clear ground-truth images
```

Then update the paths in `enhanced.ipynb`:

```python
TRAIN_INPUT  = r"path/to/dataset/train/input"
TEST_INPUT   = r"path/to/dataset/test/input"
TRAIN_VNDHR  = r"path/to/outputs/train_vndhr"
TEST_VNDHR   = r"path/to/outputs/test_vndhr"
TRAIN_TARGET = r"path/to/dataset/train/target"
```

### 5. Launch and run

```bash
jupyter notebook codes/enhanced.ipynb
```

Run cells **in order** — Stage 1 (VNDHR preprocessing) must complete before Stage 2 (U-Net training).

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 2–4, or set `IMG_SIZE = 128` |
| `FileNotFoundError` on paths | Update all `r"D:\\..."` paths in the notebook to your local paths |
| Slow VNDHR preprocessing | Reduce `max_iter` to 5 for faster (slightly lower quality) output |
| VGG perceptual loss error | Run `pip install torchvision` |
| Low PSNR during training | Verify input/target filenames are correctly matched |

---

## 👥 Contributors

<div align="center">

| | Name | GitHub | Role |
|---|---|---|---|
| <img src="https://avatars.githubusercontent.com/u/70313033?v=4" width="48" style="border-radius:50%"/> | **Tanmayee** | [@Tanmayee1802](https://github.com/Tanmayee1802) | Collaborator |
| <img src="https://github.com/karthik26-Thalari.png" width="48" style="border-radius:50%"/> | **Karthik** | [@karthik26-Thalari](https://github.com/karthik26-Thalari) | Author |

</div>

---

## 📄 License

This project is for **academic research purposes only**. If you use this work, please cite the original paper:

```bibtex
@article{vndhr2025,
  title={VNDHR: Variational Single Nighttime Image Dehazing for Enhancing
         Visibility in Intelligent Transportation Systems via Hybrid Regularization},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025}
}
```

---

## 🙌 Acknowledgements

| Resource | Role |
|---|---|
| [IEEE TITS 2025](https://ieeexplore.ieee.org/) | Original VNDHR paper & methodology |
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [scikit-image](https://scikit-image.org/) | PSNR / SSIM evaluation metrics |
| [SciPy](https://scipy.org/) | Sparse matrix PCG solver |
| [Kaggle](https://www.kaggle.com/) | GPU compute for training |

---

<div align="center">

Made with ❤️ by [karthik26-Thalari](https://github.com/karthik26-Thalari) & [Tanmayee1802](https://github.com/Tanmayee1802)

</div>
