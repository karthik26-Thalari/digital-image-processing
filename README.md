# 🌫️ VNDHR + UNet — Nighttime Image Dehazing

A **research implementation** of a two-stage nighttime image dehazing pipeline combining classical variational optimization (**VNDHR**) with a deep **U-Net** refinement model. Based on the paper:

> **"VNDHR: Variational Single Nighttime Image Dehazing for Enhancing Visibility in Intelligent Transportation Systems via Hybrid Regularization"**
> — *IEEE Transactions on Intelligent Transportation Systems (TITS), 2025*

---

## ✨ What This Project Does

Takes a **hazy nighttime image** → outputs a **clear, defogged image** using a two-stage pipeline:

1. **Stage 1 — VNDHR Preprocessing** *(classical variational method)*
   Decomposes each image into illumination (I) and reflectance (R) components using hybrid regularization (ℓp norm + weighted ℓ2 + ℓ1 TV), solving both sub-problems iteratively via Preconditioned Conjugate Gradient (PCG).

2. **Stage 2 — Deep U-Net Refinement** *(deep learning)*
   A 5-level U-Net with BatchNorm and skip connections refines the VNDHR output to recover fine details and maximize perceptual quality. Trained with MSE + L1 loss and VGG-based perceptual loss across 3 progressive stages.

---

## 📊 Results

| Method | PSNR (dB) | SSIM | Notes |
|---|---|---|---|
| VNDHR only | 19.79 | 0.8470 | Classical baseline |
| **VNDHR + UNet (ours)** | **26.01** | **0.9011** | **+6.22 dB improvement** |
| Target | 28.0 | 0.8000 | Paper target |

**Key takeaways:**
- SSIM of **0.9011 exceeds the target by 12.6%** — excellent structural preservation ✅
- PSNR of 26.01 dB is competitive with state-of-the-art on real-world haze (O-Haze: 23–25 dB, I-Haze: 24–27 dB)
- The UNet adds **+6.22 dB** PSNR and **+0.054 SSIM** over VNDHR alone, validating the two-stage approach

Evaluated on **1,000 test images** at 256×256 resolution.

---

## 🗂️ Project Structure

```
digital-image-processing-main/
│
├── codes/
│   ├── enhanced.ipynb              # Main notebook: VNDHR preprocessing + UNet training + evaluation
│   ├── research paper code.ipynb   # Experimental / earlier version notebook
│   └── raw/                        # Draft notebooks (Untitled1, Untitled2)
│
├── dataset/
│   ├── train/
│   │   ├── input/                  # Hazy training images (place your data here)
│   │   └── target/                 # Ground-truth clear images
│   └── test/
│       ├── input/                  # Hazy test images
│       └── target/                 # Ground-truth clear images
│
├── outputs/
│   ├── COMPARISON_4_PANEL/         # Side-by-side comparison images (hazy | VNDHR | UNet | GT)
│   ├── enhanced results/
│   │   ├── comparison_images/      # Enhanced comparison images
│   │   ├── dehazed_images/         # Final dehazed output images
│   │   ├── metrics_results.csv     # Per-image PSNR and SSIM scores
│   │   └── summary_report.txt      # Full evaluation summary
│   ├── VNDHR_METRICS/
│   │   ├── vndhr_detailed_metrics.csv    # Per-image VNDHR-only metrics
│   │   └── vndhr_metrics_report.txt      # VNDHR baseline comparison report
│   └── train_vndhr/                # VNDHR-preprocessed training images
│
├── trained unet model/
│   └── unet_vndhr_aggressive.pth   # Saved best model checkpoint
│
├── paper/
│   └── VNDHR_...IEEE_TITS.pdf      # Reference paper (IEEE TITS 2025)
│
└── README.md
```

---

## ⚙️ Prerequisites

- Python **3.8+**
- CUDA-compatible GPU (recommended; 6GB+ VRAM)
- Jupyter Notebook or JupyterLab

---

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/your-username/digital-image-processing.git
cd digital-image-processing
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow numpy scipy scikit-image tqdm jupyter
```

> For CPU-only:
> ```bash
> pip install torch torchvision torchaudio
> pip install opencv-python pillow numpy scipy scikit-image tqdm jupyter
> ```

### 4. Add your dataset

Place your hazy/clear image pairs into:
```
dataset/train/input/    ← hazy training images
dataset/train/target/   ← corresponding clear images
dataset/test/input/     ← hazy test images
dataset/test/target/    ← corresponding clear images
```

Then update the paths in `enhanced.ipynb`:

```python
TRAIN_INPUT  = r"path/to/dataset/train/input"
TEST_INPUT   = r"path/to/dataset/test/input"
TRAIN_VNDHR  = r"path/to/outputs/train_vndhr"
TEST_VNDHR   = r"path/to/outputs/test_vndhr"
TRAIN_TARGET = r"path/to/dataset/train/target"
```

### 5. Launch the notebook

```bash
jupyter notebook codes/enhanced.ipynb
```

Run the cells in order.

---

## 🔬 Pipeline Details

### Stage 1 — VNDHR Variational Model

The VNDHR model decomposes a hazy image **S** into:
- **I** — Illumination map (V-channel in HSV space)
- **R** — Reflectance map

Using the hybrid energy functional (paper Section III-B):

```
min  ||S - I·R||² + λ₁·||∇I||ₚ + λ₂·Wᵣ||∇R||₂² + λ₃·||∇R||₁
 I,R
```

| Parameter | Value | Role |
|---|---|---|
| λ₁ | 0.002 | ℓp norm weight for illumination |
| λ₂ | 0.0001 | Weighted ℓ2 norm for reflectance |
| λ₃ | 0.001 | ℓ1 TV norm for noise suppression |
| p | 0.65 | Fractional-order norm |
| max_iter | 10 | Optimization iterations |

Each sub-problem is solved via **Preconditioned Conjugate Gradient (PCG)** for efficiency.

### Stage 2 — U-Net Training (3-Stage Curriculum)

| Stage | Epochs | LR | Loss | Augmentation |
|---|---|---|---|---|
| 1 — Perceptual intro | 0–40 | 3e-5 | MSE + L1 + VGG (gradually added) | Light |
| 2 — Fine-tuning | 41–70 | 5e-6 | MSE + L1 (60:40) | Medium |
| 3 — Aggressive push | 71–100 | 5e-5 | MSE + L1 (60:40) | Heavy |

**Architecture:** 5-level U-Net, ~18M parameters, BatchNorm, skip connections, AdamW optimizer, mixed-precision (AMP), 4-way TTA at test time.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 2–4, or set `IMG_SIZE = 128` |
| `FileNotFoundError` on paths | Update all `r"D:\\..."` paths in the notebook to your local paths |
| Slow VNDHR preprocessing | Reduce `max_iter` to 5 for faster (slightly lower quality) output |
| VGG perceptual loss error | Make sure `torchvision` is installed: `pip install torchvision` |
| Low PSNR during training | Check that input/target pairs are correctly matched by filename |

---

## 📄 License

This project is for academic research purposes. Please cite the original paper if you use this work:

> *VNDHR: Variational Single Nighttime Image Dehazing for Enhancing Visibility in Intelligent Transportation Systems via Hybrid Regularization* — IEEE TITS 2025

---

## 🙌 Acknowledgements

- **IEEE TITS 2025** — Original VNDHR paper authors
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [scikit-image](https://scikit-image.org/) — PSNR / SSIM evaluation metrics
- [SciPy](https://scipy.org/) — Sparse matrix PCG solver
