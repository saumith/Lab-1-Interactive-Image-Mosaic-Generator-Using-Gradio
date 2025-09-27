import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import gradio as gr
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

# ---- Hugging Face dataset: hard-wired to CIFAR-100 ----
from datasets import load_dataset

HF_DATASET = "uoft-cs/cifar100"  # tiles come from here
HF_SPLIT = "train"
TILE_LIMIT = 0                # cap to keep mapping fast (adjust as needed)
BASE_TILE_SIZE = 32              # CIFAR-100 images are 32x32

# Global caches
_TILES_RAW_32: Optional[List[np.ndarray]] = None  # list of 32x32 RGB uint8 arrays
_TILE_CACHE_BY_SIZE: Dict[int, Tuple[List[np.ndarray], np.ndarray]] = {}  # size -> (tiles_resized, lab_means)

# =======================
# Image utils
# =======================
def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def center_crop_to_multiple(img: np.ndarray, cell: int) -> np.ndarray:
    h, w = img.shape[:2]
    H = (h // cell) * cell
    W = (w // cell) * cell
    top = (h - H) // 2
    left = (w - W) // 2
    return img[top:top+H, left:left+W, :]

def resize_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) == short_side:
        return img
    if h < w:
        new_h, new_w = short_side, int(w * short_side / h)
    else:
        new_h, new_w = int(h * short_side / w), short_side
    return np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32))**2))

# =======================
# CIFAR-100: load & cache base tiles (32x32)
# =======================
def _load_tiles_raw_32(limit: int = TILE_LIMIT) -> List[np.ndarray]:
    """Load 32x32 tiles (RGB uint8) from CIFAR-100 (uoft-cs/cifar100)."""
    global _TILES_RAW_32
    if _TILES_RAW_32 is not None:
        return _TILES_RAW_32

    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    tiles = []
    for i, ex in enumerate(ds):
        # CIFAR-100 column name is "img"
        if "img" not in ex:
            continue
        img: Image.Image = ex["img"].convert("RGB")
        if img.size != (BASE_TILE_SIZE, BASE_TILE_SIZE):
            img = img.resize((BASE_TILE_SIZE, BASE_TILE_SIZE), Image.BILINEAR)
        tiles.append(np.asarray(img))
        if limit > 0 and len(tiles) >= limit:
            break

    if len(tiles) == 0:
        raise gr.Error(f"No tiles loaded from {HF_DATASET}.")
    _TILES_RAW_32 = tiles
    return _TILES_RAW_32

def _average_color_lab(tile: np.ndarray) -> np.ndarray:
    lab = rgb2lab(tile / 255.0)
    return lab.reshape(-1, 3).mean(axis=0)

def _tiles_for_cell_size(cell_size: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Return (tiles_resized, tiles_lab_means) for the requested cell size.
    Caches results to avoid recomputation on each run.
    """
    if cell_size in _TILE_CACHE_BY_SIZE:
        return _TILE_CACHE_BY_SIZE[cell_size]

    raw_tiles = _load_tiles_raw_32()
    if cell_size == BASE_TILE_SIZE:
        tiles_resized = raw_tiles
    else:
        tiles_resized = [
            np.asarray(Image.fromarray(t).resize((cell_size, cell_size), Image.BILINEAR))
            for t in raw_tiles
        ]
    tiles_lab = np.array([_average_color_lab(t) for t in tiles_resized], dtype=np.float32)
    _TILE_CACHE_BY_SIZE[cell_size] = (tiles_resized, tiles_lab)
    return tiles_resized, tiles_lab

# =======================
# Grid / quantization
# =======================
def grid_mean_colors_vectorized(img: np.ndarray, cell: int) -> Tuple[np.ndarray, int, int]:
    H, W = img.shape[:2]
    assert H % cell == 0 and W % cell == 0
    r = H // cell
    c = W // cell
    v = img.reshape(r, cell, c, cell, 3).mean(axis=(1, 3))
    return v.astype(np.float32), r, c

def grid_mean_colors_loop(img: np.ndarray, cell: int) -> Tuple[np.ndarray, int, int]:
    H, W = img.shape[:2]
    r = H // cell
    c = W // cell
    out = np.zeros((r, c, 3), dtype=np.float32)
    for i in range(r):
        for j in range(c):
            patch = img[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
            out[i, j] = patch.mean(axis=(0,1))
    return out, r, c

def quantize_image_kmeans(img: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return img
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3).astype(np.float32)
    n = flat.shape[0]
    idx = np.random.choice(n, size=min(50000, n), replace=False)
    sample = flat[idx]
    km = KMeans(n_clusters=k, n_init=4, random_state=0)
    km.fit(sample)
    labels = km.predict(flat)
    centers = km.cluster_centers_.astype(np.uint8)
    quant = centers[labels].reshape(h, w, 3)
    return quant

# =======================
# Mapping: cells -> tiles
# =======================
def map_cells_to_tiles(mean_rgb: np.ndarray, tiles_lab: np.ndarray, tiles: List[np.ndarray]) -> np.ndarray:
    R, C, _ = mean_rgb.shape
    lab = rgb2lab(mean_rgb / 255.0).reshape(-1, 3).astype(np.float32)
    diff = lab[:, None, :] - tiles_lab[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nn = np.argmin(dist2, axis=1)
    th, tw = tiles[0].shape[:2]
    mosaic = np.zeros((R*th, C*tw, 3), dtype=np.uint8)
    for idx, t_idx in enumerate(nn):
        i = idx // C
        j = idx % C
        mosaic[i*th:(i+1)*th, j*tw:(j+1)*tw] = tiles[t_idx]
    return mosaic

def segment_preview(src: np.ndarray, cell: int) -> np.ndarray:
    mean_rgb, R, C = grid_mean_colors_vectorized(src, cell)
    out = np.zeros_like(src)
    for i in range(R):
        for j in range(C):
            out[i*cell:(i+1)*cell, j*cell:(j+1)*cell] = mean_rgb[i, j]
    return out.astype(np.uint8)

# =======================
# Full pipeline (tiles always from CIFAR-100)
# =======================
def build_mosaic(
    input_image: Image.Image,
    cell_size: int = 32,
    use_vectorized: bool = True,
    quant_k: int = 0,
    similarity_metric: str = "SSIM",
    preview_downscale_short_side: int = 768
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    # 1) Preprocess input
    src = pil_to_np(input_image)
    src = resize_short_side(src, preview_downscale_short_side)
    src = center_crop_to_multiple(src, cell_size)

    # 2) Optional quantization (preview only)
    _ = quantize_image_kmeans(src, quant_k) if quant_k > 0 else src

    # 3) Grid means
    t0 = time.perf_counter()
    if use_vectorized:
        mean_rgb, R, C = grid_mean_colors_vectorized(src, cell_size)
    else:
        mean_rgb, R, C = grid_mean_colors_loop(src, cell_size)
    t_grid = time.perf_counter() - t0

    # 4) Tiles from CIFAR-100 (cached & resized to cell_size)
    tiles, tiles_lab = _tiles_for_cell_size(cell_size)

    # 5) Map & build mosaic
    t1 = time.perf_counter()
    mosaic = map_cells_to_tiles(mean_rgb, tiles_lab, tiles)
    t_map = time.perf_counter() - t1

    # 6) Similarity (resize to input size for fair comparison)
    H, W = src.shape[:2]
    mosaic_rs = np.asarray(Image.fromarray(mosaic).resize((W, H), Image.BILINEAR))
    if similarity_metric == "MSE":
        score = mse(src, mosaic_rs)
        score_label = f"MSE: {score:.2f}"
    else:
        score = ssim(src, mosaic_rs, channel_axis=2, data_range=255)
        score_label = f"SSIM: {score:.4f}"

    timing = f"Grid: {t_grid*1000:.1f} ms | Mapping: {t_map*1000:.1f} ms | Total: {(t_grid+t_map)*1000:.1f} ms"
    seg_prev = segment_preview(src, cell_size)

    return (
        np_to_pil(src),
        np_to_pil(seg_prev),
        np_to_pil(mosaic_rs),
        score_label,
        timing,
        f"{R} x {C} cells (cell={cell_size}px) | tiles={len(tiles)} from {HF_DATASET}"
    )

# =======================
# Performance sweep
# =======================
def perf_sweep(input_image: Image.Image, grid_sizes: List[int] = [16, 24, 32, 40, 48, 64]):
    if input_image is None:
        return "Please provide an input image first."
    src = pil_to_np(input_image)
    src = resize_short_side(src, 768)
    rows = [["Grid(px)", "Vectorized(ms)", "Loop(ms)"]]
    for g in grid_sizes:
        img = center_crop_to_multiple(src, g)
        t0 = time.perf_counter()
        _ = grid_mean_colors_vectorized(img, g)
        v_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        _ = grid_mean_colors_loop(img, g)
        l_ms = (time.perf_counter() - t1) * 1000
        rows.append([g, f"{v_ms:.1f}", f"{l_ms:.1f}"])
    md = "| Grid(px) | Vectorized(ms) | Loop(ms) |\n|---:|---:|---:|\n"
    for r in rows[1:]:
        md += f"| {r[0]} | {r[1]} | {r[2]} |\n"
    return md

# =======================
# Gradio UI (simplified)
# =======================
EXAMPLES_DIR = Path("examples")
EXAMPLES_DIR.mkdir(exist_ok=True)
if not (EXAMPLES_DIR / "gradient1.png").exists():
    g1 = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480,1))
    grad1 = np.dstack([g1, np.flipud(g1).copy(), np.roll(g1, 160, axis=1)])
    Image.fromarray(grad1).save(EXAMPLES_DIR/"gradient1.png")

with gr.Blocks(title="Image Mosaic (CIFAR-100 tiles)", css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        f"""
        # 🧩 Image Mosaic Generator (tiles from `{HF_DATASET}`)
        - Tiles auto-loaded from **Hugging Face** dataset: `{HF_DATASET}` (split `{HF_SPLIT}`, limit {TILE_LIMIT}).  
        - Upload an image and generate a mosaic immediately.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Input image")
            gr.Examples(
                examples=[[str(EXAMPLES_DIR/"gradient1.png")]],
                inputs=[inp],
                label="Example"
            )
            cell = gr.Slider(16, 128, value=32, step=2, label="Grid cell size (px)")

            quant_k = gr.Slider(0, 24, value=0, step=1, label="Optional color quantization (k-means K)")
            similarity = gr.Radio(choices=["SSIM", "MSE"], value="SSIM", label="Similarity metric")
            vec = gr.Checkbox(value=True, label="Use vectorized NumPy (uncheck for loop baseline)")
            run = gr.Button("Generate Mosaic", variant="primary")

        with gr.Column(scale=1):
            orig = gr.Image(label="Original (cropped/resized)", interactive=False)
            seg = gr.Image(label="Segmented (cell means)", interactive=False)
            out = gr.Image(label="Mosaic", interactive=False)

            with gr.Row():
                sim_out = gr.Label(label="Similarity")
                time_out = gr.Label(label="Timing")
            meta = gr.Label(label="Grid / Tiles info")

            gr.Markdown("### Performance sweep")
            perf_btn = gr.Button("Run Performance Sweep")
            perf_table = gr.Markdown()

    run.click(
        build_mosaic,
        inputs=[inp, cell, vec, quant_k, similarity],
        outputs=[orig, seg, out, sim_out, time_out, meta]
    )
    perf_btn.click(perf_sweep, inputs=[inp], outputs=[perf_table])

if __name__ == "__main__":
    # Preload tiles at startup so first run is snappy
    try:
        _load_tiles_raw_32(TILE_LIMIT)
    except Exception as e:
        print("Warning: failed to preload tiles:", e)
    demo.launch()
