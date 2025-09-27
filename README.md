# 🧩 Interactive Image Mosaic Generator

An interactive mosaic generator that reconstructs an uploaded image using small image tiles. Built with Python, NumPy, scikit-image, scikit-learn, Hugging Face Datasets, and Gradio.

---

## 🔗 Demo

Try it out online: **[Mosaic Generator Demo](https://huggingface.co/spaces/Saumith/MosaicGeneration)**

---

## 🔍 Features

- **Real-time mosaic generation** from uploaded images
- **CIFAR-100 dataset integration** - Tiles drawn from the `uoft-cs/cifar100` dataset
- **Adjustable grid size** - Range from 16px up to 128px for different detail levels
- **Multiple similarity metrics**:
  - **MSE** (Mean Squared Error) - measures pixel-level differences
  - **SSIM** (Structural Similarity Index) - measures perceptual similarity
- **Advanced options**:
  - **Unique tiles mode** - Avoid repeat tile usage when possible
  - **Random transformations** - Flip and rotate tiles for aesthetic variety
- **Optimized implementations**:
  - **Vectorized NumPy** processing for maximum speed
  - **Loop-based** implementation as baseline comparison
- **Performance optimizations** with intelligent caching

---

## 🧱 How It Works

### 1. **Preprocessing**
Input image is resized and center-cropped to make its dimensions divisible by the chosen grid size.

### 2. **Grid Segmentation** 
The image is split into equal-sized cells, and each cell's average RGB color is computed.

### 3. **Tile Preparation**
CIFAR-100 images are scaled to match the grid cell size. Each tile's mean color is computed in LAB color space for optimal perceptual matching.

### 4. **Tile Mapping**
For each cell, the nearest tile (in LAB color space) is selected using distance metrics. Tiles can optionally be used uniquely or randomly transformed.

### 5. **Mosaic Construction**
Selected tiles are assembled into the final mosaic, which is then resized to match the original image dimensions for accurate metric evaluation.

### 6. **Quality Metrics**
- **MSE**: Measures raw pixel differences (lower is better)
- **SSIM**: Measures structural and perceptual similarity (0–1 scale, higher is better)

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/MosaicGeneration.git
cd MosaicGeneration

# Create virtual environment (recommended)
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
gradio>=4.0.0
numpy>=1.21.0
scikit-image>=0.19.0
scikit-learn>=1.0.0
datasets>=2.0.0
huggingface_hub>=0.10.0
Pillow>=9.0.0
matplotlib>=3.5.0
```

---

## ▶️ Running Locally

```bash
python app.py
```

Open your browser and navigate to the displayed address (typically `http://localhost:7860`) to access the Gradio interface.

---

## 📊 Performance & Trade-offs

### Speed Optimization
- **Vectorized implementation** is significantly faster than loop-based processing
- **Tile mapping** is the primary computational bottleneck, especially for fine grids
- **Caching mechanisms** reduce redundant calculations

### Quality vs Speed Trade-offs
| Grid Size | Detail Level | Runtime | Use Case |
|-----------|--------------|---------|----------|
| 16px | High detail | 2-5s | Artistic, high-quality output |
| 32px | Medium detail | 1-2s | Balanced quality/speed |
| 64px | Lower detail | 0.5-1s | Quick previews |
| 128px | Abstract | <0.5s | Rapid prototyping |

### Metrics Comparison
- **SSIM** provides better perceptual quality assessment than MSE
- **MSE** is useful for pixel-level accuracy measurement
- Mosaics inherently have high pixel error but can preserve structural similarity

---

## 📂 Project Structure

```
MosaicGeneration/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── examples/             # Sample input images
│   ├── landscape.jpg
│   ├── portrait.jpg
│   └── abstract.jpg
└── src/                  # Source code modules
    ├── __init__.py
    ├── mosaic_generator.py    # Core mosaic logic
    ├── tile_processor.py     # Tile preparation and caching
    └── metrics.py            # Quality evaluation metrics
```

---

## 🚀 Deployment

### Hugging Face Spaces
This application is deployed on Hugging Face Spaces for easy public access.

**Live Demo**: [Mosaic Generator Demo](https://huggingface.co/spaces/Saumith/MosaicGeneration)

### Local Deployment
For production deployment, consider:
- Using Docker containers for consistent environments
- Implementing proper error handling and logging
- Adding rate limiting for API endpoints
- Optimizing for your specific hardware configuration

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CIFAR-100 Dataset**: University of Toronto Computer Science Department
- **Hugging Face**: For hosting and dataset infrastructure
- **Gradio**: For the excellent web interface framework
- **scikit-image & scikit-learn**: For robust image processing and ML utilities

---

## 📈 Future Enhancements

- [ ] Support for custom tile datasets
- [ ] Advanced color matching algorithms
- [ ] GPU acceleration for large images
- [ ] Batch processing capabilities
- [ ] Export options (high-resolution, different formats)
- [ ] Interactive tile placement editing
- [ ] Style transfer integration

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/MosaicGeneration/issues) page
2. Create a new issue with detailed description
3. Join our [Discussions](https://github.com/your-username/MosaicGeneration/discussions) for community support

---

**⭐ Star this repository if you find it helpful!**
