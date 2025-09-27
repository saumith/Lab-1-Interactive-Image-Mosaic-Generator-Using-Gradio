# 🧩 Interactive Image Mosaic Generator

An interactive **image mosaic generator** that reconstructs an input image using small image tiles.  
Built with **Python, NumPy, scikit-image, scikit-learn, Hugging Face Datasets, and Gradio**.

![Demo](https://huggingface.co/spaces/Saumith/MosaicGeneration)  
*(example mosaic output — generated using CIFAR-100 tiles)*

---

## 🚀 Features
- Divide input images into grids (customizable cell size).
- Replace each grid cell with the closest matching **CIFAR-100 image tile**.
- Supports both **vectorized NumPy** and **loop-based** implementations.
- Choose between similarity metrics:
  - **MSE (Mean Squared Error)** — pixel-level error.
  - **SSIM (Structural Similarity Index)** — perceptual similarity.
- Optional settings:
  - Unique-tile assignment (no duplicates).
  - Random flips/rotations for variety.
- Live **Gradio interface** with adjustable parameters.
- Deployed on Hugging Face Spaces for public access.

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/MosaicGeneration.git
cd MosaicGeneration

# create a venv (recommended)
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# install requirements
pip install -r requirements.txt
▶️ Usage
Run locally:

bash
Copy code
python app.py
Gradio will launch at http://localhost:7860.
Upload an image and experiment with grid size, metrics, and settings.

🌐 Live Demo
Try the app directly on Hugging Face Spaces:
👉 Mosaic Generation on Hugging Face

📊 Performance
Vectorized NumPy is up to 20× faster than loops on fine grids.

SSIM captures perceptual similarity better than MSE.

Runtime scales with grid size:

16×16 → detailed but slow (~6s).

32×32 → balanced (~1.5s).

64×64 → fastest (<0.5s) but blockier.

📂 Project Structure
arduino
Copy code
MosaicGeneration/
│── app.py             # main Gradio app
│── requirements.txt   # dependencies
│── README.md          # project documentation
└── examples/          # sample images
