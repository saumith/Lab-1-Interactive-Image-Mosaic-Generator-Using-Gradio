#!/usr/bin/env python3



import time
import tempfile
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

from skimage.metrics import structural_similarity as ssim_metric
from skimage.color import rgb2lab
from datasets import load_dataset


# ----------------------------
# Utilities
# ----------------------------
def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img).astype(np.float32)

def np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def to_lab(arr_rgb: np.ndarray) -> np.ndarray:
    # arr in [0,255]
    return rgb2lab(arr_rgb / 255.0)

def maybe_quantize(img: Image.Image, enabled: bool, colors: int) -> Image.Image:
    if not enabled:
        return img
    # Median-cut quantization; disable dithering to avoid speckle
    return img.convert("RGB").quantize(
        colors=colors, method=Image.MEDIANCUT, dither=Image.Dither.NONE
    ).convert("RGB")

def mean_color(arr_rgb: np.ndarray) -> np.ndarray:
    # Mean in LAB for perceptual matching
    lab = to_lab(arr_rgb)
    return lab.reshape(-1, 3).mean(axis=0)


# ----------------------------
# Dataset tiles
# ----------------------------
class TileBank:
    def __init__(self):
        self.tile_images = None  # list[PIL.Image]
        self.features = None     # (N,3) mean LAB

    def load(self, sample_size: int = 2000) -> None:
        """Deterministic: take the first N images from CIFAR-100."""
        try:
            ds = load_dataset("uoft-cs/cifar100", split="train")
        except Exception:
            ds = load_dataset("cifar100", split="train")

        n = min(sample_size, len(ds))
        imgs, feats = [], []
        for i in range(n):
            rec = ds[i]
            if "img" in rec and isinstance(rec["img"], Image.Image):
                pil_img = rec["img"].convert("RGB")
            elif "image" in rec and isinstance(rec["image"], Image.Image):
                pil_img = rec["image"].convert("RGB")
            else:
                arr = rec.get("img", rec.get("image", None))
                if arr is None:
                    continue
                pil_img = Image.fromarray(np.array(arr)).convert("RGB")

            arr_rgb = pil_to_np_rgb(pil_img)
            feats.append(mean_color(arr_rgb))
            imgs.append(pil_img)

        self.tile_images = imgs
        self.features = np.vstack(feats) if feats else np.zeros((0, 3), dtype=np.float32)

    def nearest_tile_indices(self, cell_means: np.ndarray, vectorized: bool = True) -> np.ndarray:
        if self.features is None or len(self.features) == 0:
            raise RuntimeError("TileBank not loaded or empty.")

        A = cell_means.astype(np.float32)     # (K,3)
        B = self.features.astype(np.float32)  # (N,3)

        if vectorized:
            # Pairwise L2 using (a-b)^2 = a^2 + b^2 - 2ab
            A2 = (A**2).sum(axis=1, keepdims=True)     # Kx1
            B2 = (B**2).sum(axis=1, keepdims=True).T   # 1xN
            AB = A @ B.T                                # KxN
            d2 = A2 + B2 - 2 * AB
            return np.argmin(d2, axis=1)
        else:
            idxs = []
            for cm in A:
                d2 = ((B - cm) ** 2).sum(axis=1)
                idxs.append(int(np.argmin(d2)))
            return np.array(idxs, dtype=int)


# ----------------------------
# Grid helpers
# ----------------------------
def crop_to_multiple(img: Image.Image, grid_n: int) -> Image.Image:
    """Crop minimally so width/height are multiples of grid_n (ensures integral cells)."""
    w, h = img.size
    new_w = max((w // grid_n) * grid_n, grid_n)
    new_h = max((h // grid_n) * grid_n, grid_n)
    if new_w != w or new_h != h:
        img = img.crop((0, 0, new_w, new_h))
    return img

def overlay_grid(img: Image.Image, grid_n: int, line_width: int = 1) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cell_w = w // grid_n
    cell_h = h // grid_n
    for x in range(0, w + 1, cell_w):
        draw.line([(x, 0), (x, h)], fill=(255, 0, 0), width=line_width)
    for y in range(0, h + 1, cell_h):
        draw.line([(0, y), (w, y)], fill=(255, 0, 0), width=line_width)
    return img

def prepare_cells_and_means(base_img: Image.Image, grid_n: int):
    """
    Returns:
      - original RGB array (HxWx3 float32)
      - dims: (w,h,cell_w,cell_h)
      - cell_means_lab: (grid_n*grid_n, 3) mean in LAB per cell
    """
    img = base_img.convert("RGB")
    w, h = img.size
    cell_w = w // grid_n
    cell_h = h // grid_n
    arr = pil_to_np_rgb(img)  # HxWx3 in [0,255]

    lab = to_lab(arr)
    cells = lab.reshape(grid_n, cell_h, grid_n, cell_w, 3).swapaxes(1, 2)     # (grid_n,grid_n,cell_h,cell_w,3)
    means = cells.mean(axis=(2, 3)).reshape(-1, 3)                             # (grid_n*grid_n,3)
    return arr, (w, h, cell_w, cell_h), means


# ----------------------------
# Mosaic composition
# ----------------------------
def downsample_then_scale(tile_img: Image.Image, inner_px: int, target_w: int, target_h: int) -> Image.Image:
    """
    Downsample a CIFAR tile to inner_px (e.g., 8/16/24/32) to control blockiness,
    then scale up to the target cell size with NEAREST to preserve the chunky effect.
    """
    inner_px = max(1, int(inner_px))
    tiny = tile_img.resize((inner_px, inner_px), Image.BILINEAR)
    return tiny.resize((target_w, target_h), Image.NEAREST)

def compose_mosaic(tile_bank: TileBank, idxs: np.ndarray, dims: Tuple[int,int,int,int], grid_n: int, tile_px: int) -> Image.Image:
    w, h, cell_w, cell_h = dims
    out = Image.new("RGB", (w, h))
    k = 0
    for gy in range(grid_n):
        for gx in range(grid_n):
            tile_img = tile_bank.tile_images[int(idxs[k])]
            k += 1
            inner = min(tile_px, cell_w, cell_h)  # clamp ≤ cell size
            out.paste(downsample_then_scale(tile_img, inner, cell_w, cell_h), (gx * cell_w, gy * cell_h))
    return out


# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(original_rgb: np.ndarray, mosaic_rgb: np.ndarray):
    mse = float(np.mean((original_rgb - mosaic_rgb) ** 2))
    ssim_vals = []
    for c in range(3):
        ssim_vals.append(ssim_metric(original_rgb[..., c].astype(np.uint8),
                                     mosaic_rgb[..., c].astype(np.uint8),
                                     data_range=255))
    return mse, float(np.mean(ssim_vals))


# ----------------------------
# Global tilebank cache
# ----------------------------
_TILEBANKS = {}  # key: sample_size -> TileBank

def get_tilebank(sample_size: int) -> TileBank:
    key = int(sample_size)
    if key not in _TILEBANKS:
        tb = TileBank()
        tb.load(sample_size=sample_size)
        _TILEBANKS[key] = tb
    return _TILEBANKS[key]


# ----------------------------
# Gradio callback
# ----------------------------
def run_pipeline(img: Image.Image,
                 grid_size_choice: str,
                 tile_px_choice: str,
                 tile_sample_size: int,
                 quantize_on: bool,
                 quantize_colors: int,
                 show_grid_overlay: bool):

    if img is None:
        return None, None, None, None, None, None, "Please upload an image."

    grid_n = int(grid_size_choice)
    tile_px = int(tile_px_choice)  # per-tile inner resolution (px)

    # Crop for exact cell division
    base = crop_to_multiple(img.convert("RGB"), grid_n)

    # Optional quantization (before computing cell means)
    preproc = maybe_quantize(base, quantize_on, quantize_colors)

    # Segmented (grid overlay) for display
    segmented = overlay_grid(preproc, grid_n) if show_grid_overlay else preproc

    # Load/prepare tile bank
    t_load0 = time.perf_counter()
    tilebank = get_tilebank(tile_sample_size)
    t_load1 = time.perf_counter()
    load_time = t_load1 - t_load0

    # Compute cell means once (LAB vectorized)
    orig_arr, dims, means = prepare_cells_and_means(preproc, grid_n)

    # --- Vectorized pipeline ---
    t_vec0 = time.perf_counter()
    idxs_vec = tilebank.nearest_tile_indices(means, vectorized=True)
    mosaic_vec = compose_mosaic(tilebank, idxs_vec, dims, grid_n, tile_px)
    t_vec1 = time.perf_counter()
    vec_time = t_vec1 - t_vec0
    mse_vec, ssim_vec = compute_metrics(orig_arr, pil_to_np_rgb(mosaic_vec))

    # --- Loop pipeline ---
    t_loop0 = time.perf_counter()
    idxs_loop = tilebank.nearest_tile_indices(means, vectorized=False)
    mosaic_loop = compose_mosaic(tilebank, idxs_loop, dims, grid_n, tile_px)
    t_loop1 = time.perf_counter()
    loop_time = t_loop1 - t_loop0
    mse_loop, ssim_loop = compute_metrics(orig_arr, pil_to_np_rgb(mosaic_loop))

    total_time = load_time + vec_time + loop_time

    # Save mosaics to temp files for download buttons
    tmp_vec = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    mosaic_vec.save(tmp_vec.name, format="PNG")
    vec_path = tmp_vec.name

    tmp_loop = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    mosaic_loop.save(tmp_loop.name, format="PNG")
    loop_path = tmp_loop.name

    w, h, cell_w, cell_h = dims
    report = (
        f"Grid: {grid_n}×{grid_n} | Cells: {cell_w}×{cell_h}px each | Tile Size (px): {tile_px} (auto-clamped ≤ cell)\n"
        f"Tiles used: {tile_sample_size}\n"
        f"Quantization: {'ON' if quantize_on else 'OFF'}"
        f"{f' ({quantize_colors} colors)' if quantize_on else ''}\n"
        f"Tile load/precompute: {load_time:.3f}s | Total (all): {total_time:.3f}s\n"
        f"[Vectorized]  Time: {vec_time:.3f}s | MSE: {mse_vec:.2f} | SSIM: {ssim_vec:.4f}\n"
        f"[Loop]       Time: {loop_time:.3f}s | MSE: {mse_loop:.2f} | SSIM: {ssim_loop:.4f}"
    )

    # Return both mosaics AND the two file paths for the download buttons
    return base, segmented, mosaic_vec, mosaic_loop, vec_path, loop_path, report


# ----------------------------
# Gradio UI
# ----------------------------
def build_demo():
    with gr.Blocks(title="Interactive Image Mosaic Generator") as demo:
        gr.Markdown(
            """
            # Interactive Image Mosaic Generator
            - **Grid size = number of tiles per side** (e.g., 32 ⇒ 32×32).
            - **Tile Size (px)** = internal resolution per tile (downsample then scale), usually **smaller** than the cell size.
            - Tiles: **Hugging Face `uoft-cs/cifar100`** (fallback: `cifar100`).
            - **Both implementations** run each time: Vectorized & Loop (reference).
            - Optional **color quantization** before analysis.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="Upload Image")

                grid_size = gr.Radio(
                    choices=["16", "32", "64", "128"],
                    value="32",
                    label="Grid size (cells per side)"
                )

                tile_px = gr.Radio(
                    choices=["8", "16", "24", "32"],
                    value="16",
                    label="Tile Size (px, ≤ cell size)"
                )

                tile_sample_size = gr.Slider(
                    minimum=256,
                    maximum=10000,
                    step=256,
                    value=2048,
                    label="Number of tiles to sample from CIFAR-100"
                )

                with gr.Accordion("Preprocessing: Color Quantization (optional)", open=False):
                    quantize_on = gr.Checkbox(value=False, label="Apply color quantization")
                    quantize_colors = gr.Slider(
                        minimum=8, maximum=128, step=8, value=32,
                        label="Quantization palette size (colors)"
                    )

                show_grid = gr.Checkbox(value=True, label="Show grid overlay on segmented preview")
                run_btn = gr.Button("Generate Mosaic", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Original (cropped)"):
                    img_orig = gr.Image(label="Original (cropped to grid multiple)")
                with gr.Tab("Segmented"):
                    img_seg = gr.Image(label="Segmented (grid overlay / preprocessed)")
                with gr.Tab("Mosaic — Vectorized (Fast)"):
                    img_vec = gr.Image(label="Vectorized Mosaic")
                    download_vec = gr.DownloadButton(label="⬇️ Download Vectorized Mosaic")
                with gr.Tab("Mosaic — Loop (Reference)"):
                    img_loop = gr.Image(label="Loop Mosaic")
                    download_loop = gr.DownloadButton(label="⬇️ Download Loop Mosaic")
                report = gr.Textbox(label="Metrics & Timing", lines=8)

        run_btn.click(
            fn=run_pipeline,
            inputs=[img_in, grid_size, tile_px, tile_sample_size, quantize_on, quantize_colors, show_grid],
            outputs=[img_orig, img_seg, img_vec, img_loop, download_vec, download_loop, report]
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
