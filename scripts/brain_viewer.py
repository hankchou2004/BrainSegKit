"""
Brain Viewer — OASIS-1 / FreeSurfer 41-class aseg
===================================================
Usage:
    python brain_viewer.py [--split train|val|test] [--dataset-root PATH]

Controls:
  ← / →     : switch case
  A         : all labels ON
  N         : all labels OFF
  H         : toggle hippocampus (L+R)
  V         : toggle ventricles
  C         : toggle cortex (L+R)
  1         : White Matter only
  2         : Gray Matter only
  3         : CSF / Ventricles only
  Scroll    : zoom in/out (hover over a view)
  Click     : toggle a label in the panel

Crosshairs:
  Magenta  = Sagittal X    Cyan = Coronal Y    Yellow = Axial Z
"""

import argparse
import glob
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # WSLg / X11 compatible
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Slider, Button

# ── dark theme ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "axes.facecolor":   "#0d1117",
    "figure.facecolor": "#0d1117",
    "text.color":       "#e6edf3",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "font.family":      "monospace",
})

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Brain MRI viewer — FreeSurfer aseg 41 classes")
parser.add_argument(
    "--split", choices=["train", "val", "test"], default="train",
    help="Which dataset split to view (default: train)",
)
parser.add_argument(
    "--dataset-root", default="/home/hank/medical_segmention/dataset",
    help="Path to dataset/ directory (default: /home/hank/medical_segmention/dataset)",
)
args = parser.parse_args()

SPLIT       = args.split
DATASET_ROOT = Path(args.dataset_root)
SPLITS_JSON  = DATASET_ROOT / "splits.json"
FS_ROOT      = DATASET_ROOT / "freesurfer"

# Choose imagesTr/labelsTr vs imagesTs/labelsTs
if SPLIT in ("train", "val"):
    image_dir = FS_ROOT / "imagesTr"
    label_dir = FS_ROOT / "labelsTr"
else:
    image_dir = FS_ROOT / "imagesTs"
    label_dir = FS_ROOT / "labelsTs"

# Load subject IDs for the requested split from splits.json
with open(SPLITS_JSON) as f:
    split_ids: list[str] = json.load(f)["splits"][SPLIT]

image_files = [str(image_dir / f"{sid}_0000.nii.gz") for sid in split_ids
               if (image_dir / f"{sid}_0000.nii.gz").exists()]
label_files = [str(label_dir / f"{sid}.nii.gz")      for sid in split_ids
               if (label_dir / f"{sid}.nii.gz").exists()]

# Align lengths (label may be missing for some subjects)
if len(label_files) != len(image_files):
    label_files = [None] * len(image_files)

print(f"Split  : {SPLIT}")
print(f"Images : {len(image_files)}")
print(f"Labels : {len(label_files)}")

# ── FreeSurfer aseg label dict (remapped indices 1–40) ────────────────────────
# Source labels: [2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,30,
#                 41,42,43,44,46,47,49,50,51,52,53,54,58,60,62,72,78,79,81,82,85]
# → remapped to consecutive indices 1–40 (0 = background, not shown)
LABEL_DICT = {
    # ── Left hemisphere ──────────────────────────────
    1:  "Left Cerebral WM",
    2:  "Left Cerebral Cortex",
    3:  "Left Lateral Ventricle",
    4:  "Left Inf Lateral Ventricle",
    5:  "Left Cerebellum WM",
    6:  "Left Cerebellum Cortex",
    7:  "Left Thalamus",
    8:  "Left Caudate",
    9:  "Left Putamen",
    10: "Left Pallidum",
    14: "Left Hippocampus",
    15: "Left Amygdala",
    17: "Left Accumbens",
    18: "Left VentralDC",
    19: "Left Vessel",
    # ── Midline / bilateral ──────────────────────────
    11: "3rd Ventricle",
    12: "4th Ventricle",
    13: "Brain Stem",
    16: "CSF",
    35: "5th Ventricle",
    36: "WM Hypointensities",
    37: "Non-WM Hypointensities",
    38: "Left WM Hypointensities",
    39: "Right WM Hypointensities",
    40: "Optic Chiasm",
    # ── Right hemisphere ─────────────────────────────
    20: "Right Cerebral WM",
    21: "Right Cerebral Cortex",
    22: "Right Lateral Ventricle",
    23: "Right Inf Lateral Ventricle",
    24: "Right Cerebellum WM",
    25: "Right Cerebellum Cortex",
    26: "Right Thalamus",
    27: "Right Caudate",
    28: "Right Putamen",
    29: "Right Pallidum",
    30: "Right Hippocampus",
    31: "Right Amygdala",
    32: "Right Accumbens",
    33: "Right VentralDC",
    34: "Right Vessel",
}

_LEFT_KEYS    = [1,2,3,4,5,6,7,8,9,10,14,15,17,18,19]
_MIDLINE_KEYS = [11,12,13,16,35,36,37,38,39,40]
_RIGHT_KEYS   = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

PANEL_ORDER = []
for g_name, g_keys in [("LEFT", _LEFT_KEYS), ("MIDLINE", _MIDLINE_KEYS), ("RIGHT", _RIGHT_KEYS)]:
    PANEL_ORDER.append(("header", g_name))
    for k in g_keys:
        PANEL_ORDER.append(("label", k))

N_LABELS  = len(LABEL_DICT)
N_HEADERS = 3

# perceptually-spread palette via golden-angle hue
COLORS: dict[int, np.ndarray] = {}
for lbl in LABEL_DICT:
    h = (lbl * 137.508) % 360 / 360.0
    s = min(0.65 + (lbl % 5) * 0.06, 1.0)
    v = min(0.82 + (lbl % 3) * 0.05, 1.0)
    COLORS[lbl] = hsv_to_rgb([h, s, v])

# ── viewer state ──────────────────────────────────────────────────────────────
current_case  = 0
active_labels = set(LABEL_DICT.keys())
image = label_vol = None
sx_mm = sy_mm = sz_mm = 1.0

# ── figure layout ─────────────────────────────────────────────────────────────
PANEL_FRAC = 0.175
VIEW_LEFT  = PANEL_FRAC + 0.012
VIEW_RIGHT = 0.997
VIEWS_W    = VIEW_RIGHT - VIEW_LEFT
EACH_W     = VIEWS_W / 3.0
TOP        = 0.955
BOTTOM     = 0.19
GAP        = 0.005
SL_LEFT_X  = VIEW_LEFT + 0.04
SL_W       = VIEWS_W - 0.05
SL_H       = 0.025
SL_BOT     = 0.015
BTN_X      = 0.004
BTN_W      = PANEL_FRAC - 0.01
BTN_H      = 0.032

fig = plt.figure(figsize=(23, 10), facecolor="#0d1117")
try:
    fig.canvas.manager.set_window_title(f"Brain Viewer — {SPLIT}")
except Exception:
    pass

panel_ax = fig.add_axes([0.003, BOTTOM, PANEL_FRAC - 0.006, TOP - BOTTOM])
panel_ax.set_facecolor("#161b22")
panel_ax.set_xticks([]); panel_ax.set_yticks([])
for sp in panel_ax.spines.values():
    sp.set_color("#30363d")

ax_sag = fig.add_axes([VIEW_LEFT,            BOTTOM, EACH_W - GAP, TOP - BOTTOM])
ax_cor = fig.add_axes([VIEW_LEFT +   EACH_W, BOTTOM, EACH_W - GAP, TOP - BOTTOM])
ax_axi = fig.add_axes([VIEW_LEFT + 2*EACH_W, BOTTOM, EACH_W - GAP, TOP - BOTTOM])
for ax in (ax_sag, ax_cor, ax_axi):
    ax.set_facecolor("#0d1117")
    ax.set_xticks([]); ax.set_yticks([])

sl_sag_ax = fig.add_axes([SL_LEFT_X, SL_BOT + 0.10, SL_W, SL_H], facecolor="#161b22")
sl_cor_ax = fig.add_axes([SL_LEFT_X, SL_BOT + 0.06, SL_W, SL_H], facecolor="#161b22")
sl_axi_ax = fig.add_axes([SL_LEFT_X, SL_BOT + 0.02, SL_W, SL_H], facecolor="#161b22")

SLIDER_KW = dict(
    color="#238636", track_color="#21262d",
    handle_style={"facecolor": "#58a6ff", "edgecolor": "#58a6ff", "size": 8},
)
sx = Slider(sl_sag_ax, "Sagittal", 0, 1, valinit=0, valstep=1, **SLIDER_KW)
sy = Slider(sl_cor_ax, "Coronal",  0, 1, valinit=0, valstep=1, **SLIDER_KW)
sz = Slider(sl_axi_ax, "Axial",    0, 1, valinit=0, valstep=1, **SLIDER_KW)
for sl in (sx, sy, sz):
    sl.label.set_color("#8b949e")
    sl.valtext.set_color("#58a6ff")


def _rebuild_slider(host_ax, label, dim, init_val):
    host_ax.clear()
    host_ax.set_facecolor("#161b22")
    sl = Slider(host_ax, label, 0, dim - 1, valinit=init_val, valstep=1, **SLIDER_KW)
    sl.label.set_color("#8b949e")
    sl.valtext.set_color("#58a6ff")
    return sl


btns = []
for by, bl in [(SL_BOT + 0.10, "All ON"), (SL_BOT + 0.06, "All OFF"),
               (SL_BOT + 0.02, f"Split: {SPLIT}")]:
    bax = fig.add_axes([BTN_X, by, BTN_W, BTN_H], facecolor="#21262d")
    b   = Button(bax, bl, color="#21262d", hovercolor="#30363d")
    b.label.set_color("#e6edf3"); b.label.set_fontsize(8)
    btns.append(b)
btn_all_on, btn_all_off, btn_split = btns

# ── panel ─────────────────────────────────────────────────────────────────────
ROW_H  = 1.0
HDR_H  = 1.3
PANEL_TOTAL_H = N_LABELS * ROW_H + N_HEADERS * HDR_H + 1.5
SWATCH_X = 0.02; SWATCH_W = 0.07
CHECK_X  = 0.125
TEXT_X   = 0.185
TEXT_FS  = 5.9
_panel_items: dict[int, dict] = {}


def rebuild_panel():
    panel_ax.clear()
    panel_ax.set_facecolor("#161b22")
    panel_ax.set_xticks([]); panel_ax.set_yticks([])
    panel_ax.set_xlim(0, 1)
    panel_ax.set_ylim(0, PANEL_TOTAL_H)
    for sp in panel_ax.spines.values():
        sp.set_color("#30363d")

    y = PANEL_TOTAL_H - 0.7
    for kind, val in PANEL_ORDER:
        if kind == "header":
            panel_ax.text(0.5, y, f"── {val} ──", ha="center", va="center",
                          fontsize=5.5, color="#58a6ff", style="italic",
                          transform=panel_ax.transData)
            y -= HDR_H
            continue

        lbl    = val
        active = lbl in active_labels
        col    = COLORS[lbl]
        alpha  = 1.0 if active else 0.20

        sw = mpatches.FancyBboxPatch(
            (SWATCH_X, y - 0.38), SWATCH_W, 0.72,
            boxstyle="round,pad=0.02",
            facecolor=col, edgecolor="none", alpha=alpha,
            transform=panel_ax.transData, clip_on=True,
        )
        panel_ax.add_patch(sw)
        panel_ax.text(CHECK_X, y + 0.05,
                      "✓" if active else "·",
                      fontsize=7, color="#58a6ff" if active else "#30363d",
                      va="center", ha="center", transform=panel_ax.transData)
        txt = panel_ax.text(TEXT_X, y + 0.05,
                            f"{lbl:>2}  {LABEL_DICT[lbl]}",
                            fontsize=TEXT_FS,
                            color="#dde4ee" if active else "#3d444d",
                            va="center", ha="left", transform=panel_ax.transData,
                            picker=True)
        txt._brain_label = lbl
        _panel_items[lbl] = {"text": txt, "swatch": sw}
        y -= ROW_H

    panel_ax.set_title("LABELS", color="#8b949e", fontsize=7, pad=3, loc="left")
    fig.canvas.draw_idle()


def on_panel_pick(event):
    if not hasattr(event.artist, "_brain_label"):
        return
    active_labels.symmetric_difference_update({event.artist._brain_label})
    rebuild_panel()
    update(None)

fig.canvas.mpl_connect("pick_event", on_panel_pick)

# ── overlay ───────────────────────────────────────────────────────────────────
def make_overlay(seg: np.ndarray) -> np.ndarray:
    h, w = seg.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for lbl in active_labels:
        m = seg == lbl
        if m.any():
            rgba[m, :3] = COLORS[lbl]
            rgba[m,  3] = 0.55
    return rgba

# ── crosshairs ────────────────────────────────────────────────────────────────
COL_SAG = "#ff79c6"
COL_COR = "#8be9fd"
COL_AXI = "#f1fa8c"
LW = 0.9; ALPHA = 0.85


def _crosshair(ax, h_row, v_col, h_color, v_color, nr, nc):
    h_row = int(np.clip(h_row, 0, nr - 1))
    v_col = int(np.clip(v_col, 0, nc - 1))
    ax.axhline(h_row, color=h_color, lw=LW, alpha=ALPHA, ls="--")
    ax.axvline(v_col, color=v_color, lw=LW, alpha=ALPHA, ls="--")
    ax.plot(v_col, h_row, "+", color="white", ms=9, mew=1.3, alpha=0.9, zorder=10)

# ── main update ───────────────────────────────────────────────────────────────
def update(val):
    if image is None:
        return

    xi = int(sx.val)
    yi = int(sy.val)
    zi = int(sz.val)
    nx, ny, nz = image.shape

    for ax in (ax_sag, ax_cor, ax_axi):
        ax.clear()
        ax.set_facecolor("#0d1117")
        ax.set_xticks([]); ax.set_yticks([])

    # Sagittal
    sag_img = np.rot90(image    [xi, :, :])
    sag_seg = np.rot90(label_vol[xi, :, :])
    asp_sag = sz_mm / sy_mm
    ax_sag.imshow(sag_img, cmap="gray", aspect=asp_sag, interpolation="bilinear")
    ax_sag.imshow(make_overlay(sag_seg), aspect=asp_sag, interpolation="nearest")
    _crosshair(ax_sag, nz - 1 - zi, yi, COL_AXI, COL_COR, nz, ny)
    ax_sag.set_title(f"Sagittal   x = {xi}", fontsize=8.5, color="#8b949e", pad=3)

    # Coronal
    cor_img = np.rot90(image    [:, yi, :])
    cor_seg = np.rot90(label_vol[:, yi, :])
    asp_cor = sz_mm / sx_mm
    ax_cor.imshow(cor_img, cmap="gray", aspect=asp_cor, interpolation="bilinear")
    ax_cor.imshow(make_overlay(cor_seg), aspect=asp_cor, interpolation="nearest")
    _crosshair(ax_cor, nz - 1 - zi, xi, COL_AXI, COL_SAG, nz, nx)
    ax_cor.set_title(f"Coronal    y = {yi}", fontsize=8.5, color="#8b949e", pad=3)

    # Axial
    axi_img = np.rot90(image    [:, :, zi])
    axi_seg = np.rot90(label_vol[:, :, zi])
    asp_axi = sy_mm / sx_mm
    ax_axi.imshow(axi_img, cmap="gray", aspect=asp_axi, interpolation="bilinear")
    ax_axi.imshow(make_overlay(axi_seg), aspect=asp_axi, interpolation="nearest")
    _crosshair(ax_axi, ny - 1 - yi, xi, COL_COR, COL_SAG, ny, nx)
    ax_axi.set_title(f"Axial      z = {zi}", fontsize=8.5, color="#8b949e", pad=3)

    # legend
    shown   = sorted(active_labels)[:20]
    patches = [mpatches.Patch(color=COLORS[l], label=f"{l:>2} {LABEL_DICT[l][:18]}")
               for l in shown]
    if patches:
        ax_axi.legend(handles=patches, loc="lower right",
                      fontsize=4.0, framealpha=0.6,
                      facecolor="#0d1117", edgecolor="#30363d",
                      labelcolor="#dde4ee", ncol=1, handlelength=1.1,
                      borderpad=0.4, labelspacing=0.25)

    # crosshair colour key
    fig.texts = [t for t in fig.texts if not getattr(t, "_ch_key", False)]
    t = fig.text(VIEW_LEFT, BOTTOM - 0.025,
                 f"  ╌╌ Coronal Y   ╌╌ Axial Z   ╌╌ Sagittal X   ✛ intersection",
                 fontsize=6, color="#8b949e")
    t._ch_key = True

    has_label = label_files[current_case] is not None
    fig.suptitle(
        f"Brain Viewer  │  [{SPLIT.upper()}]  Case {current_case + 1} / {len(image_files)}"
        f"  │  {len(active_labels)} labels active"
        f"{'  │  no labels' if not has_label else ''}",
        fontsize=9.5, color="#58a6ff",
        y=0.992, x=VIEW_LEFT + VIEWS_W / 2,
    )
    fig.canvas.draw_idle()

# ── load case ─────────────────────────────────────────────────────────────────
def load_case(idx):
    global image, label_vol, sx_mm, sy_mm, sz_mm
    global sx, sy, sz

    nii           = nib.load(image_files[idx])
    image         = nii.get_fdata()
    sx_mm, sy_mm, sz_mm = nii.header.get_zooms()[:3]

    if label_files[idx] is not None:
        label_vol = nib.load(label_files[idx]).get_fdata().astype(np.int32)
    else:
        label_vol = np.zeros_like(image, dtype=np.int32)

    nx, ny, nz = image.shape
    sx = _rebuild_slider(sl_sag_ax, f"Sagittal [0-{nx-1}]", nx, nx // 2)
    sy = _rebuild_slider(sl_cor_ax, f"Coronal  [0-{ny-1}]", ny, ny // 2)
    sz = _rebuild_slider(sl_axi_ax, f"Axial    [0-{nz-1}]", nz, nz // 2)
    sx.on_changed(update)
    sy.on_changed(update)
    sz.on_changed(update)
    update(None)

# ── button callbacks ──────────────────────────────────────────────────────────
def cb_all_on(event):
    active_labels.update(LABEL_DICT.keys())
    rebuild_panel(); update(None)

def cb_all_off(event):
    active_labels.clear()
    rebuild_panel(); update(None)

btn_all_on.on_clicked(cb_all_on)
btn_all_off.on_clicked(cb_all_off)

# ── keyboard ──────────────────────────────────────────────────────────────────
def on_key(event):
    global current_case, active_labels

    if event.key == "right":
        current_case = (current_case + 1) % len(image_files)
        load_case(current_case)
    elif event.key == "left":
        current_case = (current_case - 1) % len(image_files)
        load_case(current_case)
    elif event.key == "a":
        cb_all_on(None)
    elif event.key == "n":
        cb_all_off(None)
    elif event.key == "h":
        # Hippocampus: idx 14 (Left), 30 (Right)
        for l in (14, 30):
            active_labels.symmetric_difference_update({l})
        rebuild_panel(); update(None)
    elif event.key == "v":
        # Ventricles: 3, 4, 11, 12, 16, 22, 23, 35
        for l in (3, 4, 11, 12, 16, 22, 23, 35):
            active_labels.symmetric_difference_update({l})
        rebuild_panel(); update(None)
    elif event.key == "c":
        # Cortex: 2 (Left Cerebral Cortex), 21 (Right Cerebral Cortex)
        for l in (2, 21):
            active_labels.symmetric_difference_update({l})
        rebuild_panel(); update(None)
    elif event.key == "1":
        # White Matter
        active_labels = {1, 5, 20, 24}
        rebuild_panel(); update(None)
    elif event.key == "2":
        # Gray Matter
        active_labels = {2, 6, 7, 8, 9, 10, 14, 15, 17,
                         21, 25, 26, 27, 28, 29, 30, 31, 32}
        rebuild_panel(); update(None)
    elif event.key == "3":
        # CSF / Ventricles
        active_labels = {3, 4, 11, 12, 16, 22, 23, 35}
        rebuild_panel(); update(None)

fig.canvas.mpl_connect("key_press_event", on_key)

# ── scroll zoom ───────────────────────────────────────────────────────────────
def on_scroll(event):
    ax = event.inaxes
    if ax not in (ax_sag, ax_cor, ax_axi):
        return
    f  = 0.85 if event.button == "up" else 1.15
    cx = np.mean(ax.get_xlim()); cy = np.mean(ax.get_ylim())
    rx = (ax.get_xlim()[1] - ax.get_xlim()[0]) * f / 2
    ry = (ax.get_ylim()[1] - ax.get_ylim()[0]) * f / 2
    ax.set_xlim(cx - rx, cx + rx)
    ax.set_ylim(cy - ry, cy + ry)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("scroll_event", on_scroll)

# ── start ─────────────────────────────────────────────────────────────────────
rebuild_panel()
load_case(current_case)

print(f"\n── Brain Viewer  [{SPLIT.upper()}]  {len(image_files)} cases ──")
print("  ← / →   : switch case")
print("  A / N   : all labels ON / OFF")
print("  H       : hippocampus (L+R)")
print("  V       : ventricles")
print("  C       : cortex (L+R)")
print("  1       : White Matter")
print("  2       : Gray Matter")
print("  3       : CSF / Ventricles")
print("  Scroll  : zoom in/out on a view")
print("  Click   : toggle label in panel")
print("──────────────────────────────────────\n")

plt.show()
