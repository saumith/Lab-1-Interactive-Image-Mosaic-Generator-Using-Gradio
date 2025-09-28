# 🧩 Interactive Image Mosaic Generator

An interactive mosaic generator that reconstructs an uploaded image using small image tiles from **CIFAR-100**. Built with **Python, NumPy, scikit-image, Hugging Face Datasets, and Gradio**.

---

## 🔗 Demo

Try it out online: **[Mosaic Generator Demo](https://huggingface.co/spaces/Saumith/MosaicGeneration)**

---

## 🔍 Features

- **Grid sizes**: 16, 32, 64, 128 (cells per side — not pixels)
- **Tile size (px)**: Downsample CIFAR tiles (8/16/24/32) then scale to the grid cell size for a chunky effect
- **CIFAR-100 dataset integration** — tiles drawn from `uoft-cs/cifar100` (with fallback to `cifar100`)
- **Both implementations always run**:
  - **Vectorized NumPy** (fast)
  - **Loop-based** (reference)
- **Similarity metrics**:
  - **MSE** (Mean Squared Error)
  - **SSIM** (Structural Similarity Index)
- **Optional preprocessing**:
  - Input **color quantization** (median cut) to simplify palette
- **Usability**:
  - **Grid overlay preview**
  - **Download buttons** for mosaics (PNG)

---

## 🧱 How It Works

1. **Preprocessing**  
   - Image is cropped so dimensions are divisible by the chosen grid size.  
   - Optional color quantization reduces palette size.

2. **Grid Segmentation**  
   - Image is split into cells. Each cell's mean LAB color is computed.

3. **Tile Preparation**  
   - CIFAR-100 tiles are resized.  
   - Each tile's mean LAB color is precomputed.

4. **Tile Mapping**  
   - Each cell is matched to its nearest tile in LAB space.  
   - Both vectorized and loop implementations are run.

5. **Mosaic Construction**  
   - Tiles are assembled into the mosaic.  
   - Tile size (px) controls blockiness.  
   - Results are downloadable.

6. **Quality Metrics**  
   - **MSE**: pixel-wise difference (lower = better).  
   - **SSIM**: perceptual similarity (0–1 scale, higher = better).

---

## 📦 Installation

### Prerequisites
- Python 3.9–3.11 recommended  
- pip package manager  

### Setup
```bash
# Clone the repository
git clone git@github.com:saumith/Lab-1-Interactive-Image-Mosaic-Generator-Using-Gradio.git
cd Lab-1-Interactive-Image-Mosaic-Generator-Using-Gradio

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Running Locally

```bash
python app.py
```

Open the Gradio link (usually http://127.0.0.1:7860) to use the app.

---

## 📊 Performance & Trade-offs

See LAB1-REPORT.pdf for detailed metrics (MSE, SSIM, runtimes).

**Key insights:**
- Vectorization is 2–4× faster than loop-based processing.
- Tile mapping dominates runtime, especially for small grids.

**Trade-off:**
- **16×16** → Higher SSIM (best structure), slower
- **32×32** → Best balance (moderate runtime + acceptable quality)
- **64×64** → Fastest, but lower SSIM (blocky)

---

## 📂 Project Structure

```
Lab-1-Interactive-Image-Mosaic-Generator-Using-Gradio/
├── app.py             # Main Gradio app
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── LAB1-REPORT.pdf    # Performance report with results
```

---

## 🚀 Deployment

### Hugging Face Spaces
This app is live on Hugging Face Spaces:
👉 [Mosaic Generator Demo](https://huggingface.co/spaces/Saumith/MosaicGeneration)

### Local Deployment
For reproducibility:
- Use runtime.txt (e.g., python-3.10) if deploying to Hugging Face
- Enable caching to speed up CIFAR-100 dataset load
- Vectorized implementation recommended for interactive use
