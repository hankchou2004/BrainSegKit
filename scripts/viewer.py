"""
MedSeg Viewer — OASIS-1 (brain MRI, 41 classes) / BTCV (abdominal CT, 14 classes).

Usage (interactive, requires display / WSLg):
    python scripts/viewer.py --dataset oasis1 [--split train|val|test]
    python scripts/viewer.py --dataset btcv   [--split train|val|test]

Usage (SSH / headless — save PNG):
    python scripts/viewer.py --dataset oasis1 --save [--case 0] [--save-dir ./viewer_out]
    python scripts/viewer.py --dataset btcv   --save [--case 0]

Controls:
  ← / →   switch case          A / N   all labels ON / OFF
  Click   toggle label          Scroll  zoom in/out (hover a view)

OASIS-1 shortcuts:
  H   Hippocampus (L+R)    V   Ventricles    C   Cortex (L+R)
  1   White Matter only    2   Gray Matter   3   CSF / Ventricles

BTCV shortcuts:
  L   Liver    S   Spleen    K   Kidneys (L+R)    P   Pancreas    A   Aorta
  1   Solid organs (Spleen/Kidneys/Liver)
  2   Vascular (Aorta/IVC/Portal Vein)
  3   GI tract (Esophagus/Stomach/Gallbladder)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ── argument parsing (before matplotlib so backend can be set) ─────────────
parser = argparse.ArgumentParser(description="MedSeg Viewer")
parser.add_argument("--dataset",      choices=["oasis1", "btcv"], default="oasis1")
parser.add_argument("--split",        choices=["train", "val", "test"], default="train")
parser.add_argument("--dataset-root", default="/home/hank/medical_segmention/dataset")
parser.add_argument("--save",         action="store_true",
                    help="Headless: save PNG instead of opening GUI")
parser.add_argument("--case",         type=int, default=0,
                    help="Case index in --save mode (default: 0)")
parser.add_argument("--save-dir",     default="./viewer_out")
args = parser.parse_args()

import matplotlib
matplotlib.use("Agg" if args.save else "TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Slider, Button

import nibabel as nib
import numpy as np

# ── dark theme ─────────────────────────────────────────────────────────────
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


# ── dataset config ──────────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    name:         str
    label_dict:   dict[int, str]
    panel_groups: list[tuple[str, list[int]]]          # [(group_name, [label_ids])]
    shortcuts:    dict[str, tuple[str, set[int]]]      # key → (description, label_set)
    splits_json:  Callable[[Path], Path]               # dataset_root → splits.json
    img_path:     Callable[[Path, str, str], Path]     # root, sid, split → image path
    lbl_path:     Callable[[Path, str, str], Optional[Path]]  # root, sid, split → label path


# ── OASIS-1 config ──────────────────────────────────────────────────────────
_OASIS_LABELS: dict[int, str] = {
    # Left hemisphere
     2: "Left Cerebral WM",         3: "Left Cerebral Cortex",
     4: "Left Lateral Ventricle",   5: "Left Inf Lat Ventricle",
     7: "Left Cerebellum WM",       8: "Left Cerebellum Cortex",
    10: "Left Thalamus",           11: "Left Caudate",
    12: "Left Putamen",            13: "Left Pallidum",
    17: "Left Hippocampus",        18: "Left Amygdala",
    26: "Left Accumbens",          28: "Left VentralDC",
    30: "Left Vessel",
    # Midline / bilateral
    14: "3rd Ventricle",           15: "4th Ventricle",
    16: "Brain Stem",              24: "CSF",
    72: "5th Ventricle",           78: "WM Hypointensities",
    79: "Non-WM Hypointensities",  81: "Left WM Hypo",
    82: "Right WM Hypo",           85: "Optic Chiasm",
    # Right hemisphere
    41: "Right Cerebral WM",       42: "Right Cerebral Cortex",
    43: "Right Lateral Ventricle", 44: "Right Inf Lat Ventricle",
    46: "Right Cerebellum WM",     47: "Right Cerebellum Cortex",
    49: "Right Thalamus",          50: "Right Caudate",
    51: "Right Putamen",           52: "Right Pallidum",
    53: "Right Hippocampus",       54: "Right Amygdala",
    58: "Right Accumbens",         60: "Right VentralDC",
    62: "Right Vessel",
}
_OASIS_GROUPS = [
    ("LEFT",    [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30]),
    ("MIDLINE", [14, 15, 16, 24, 72, 78, 79, 81, 82, 85]),
    ("RIGHT",   [41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62]),
]
_OASIS_SHORTCUTS = {
    "h": ("Hippocampus L+R",  {17, 53}),
    "v": ("Ventricles",       {4, 5, 14, 15, 43, 44, 72}),
    "c": ("Cortex L+R",       {3, 42}),
    "1": ("White Matter",     {2, 7, 41, 46}),
    "2": ("Gray Matter",      {3, 8, 10, 11, 12, 13, 17, 18, 26,
                               42, 47, 49, 50, 51, 52, 53, 54, 58}),
    "3": ("CSF / Ventricles", {4, 5, 14, 15, 24, 43, 44, 72}),
}

OASIS1 = DatasetConfig(
    name         = "OASIS-1",
    label_dict   = _OASIS_LABELS,
    panel_groups = _OASIS_GROUPS,
    shortcuts    = _OASIS_SHORTCUTS,
    splits_json  = lambda r: r / "oasis1_freesurfer" / "splits.json",
    img_path     = lambda r, sid, sp: (
        r / "oasis1_freesurfer"
        / ("imagesTr" if sp in ("train", "val") else "imagesTs")
        / f"{sid}_0000.nii.gz"
    ),
    lbl_path     = lambda r, sid, sp: (
        r / "oasis1_freesurfer"
        / ("labelsTr" if sp in ("train", "val") else "labelsTs")
        / f"{sid}.nii.gz"
    ),
)


# ── BTCV config ─────────────────────────────────────────────────────────────
_BTCV_LABELS: dict[int, str] = {
     1: "Spleen",          2: "Right Kidney",    3: "Left Kidney",
     4: "Gallbladder",     5: "Esophagus",        6: "Liver",
     7: "Stomach",         8: "Aorta",            9: "Inf Vena Cava",
    10: "Portal Vein",    11: "Pancreas",         12: "Right Adrenal",
    13: "Left Adrenal",
}
_BTCV_GROUPS = [
    ("ABDOMINAL ORGANS", list(_BTCV_LABELS.keys())),
]
_BTCV_SHORTCUTS = {
    "l": ("Liver",         {6}),
    "s": ("Spleen",        {1}),
    "k": ("Kidneys L+R",   {2, 3}),
    "p": ("Pancreas",      {11}),
    "a": ("Aorta",         {8}),
    "1": ("Solid organs",  {1, 2, 3, 6}),
    "2": ("Vascular",      {8, 9, 10}),
    "3": ("GI tract",      {4, 5, 7}),
}

BTCV = DatasetConfig(
    name         = "BTCV",
    label_dict   = _BTCV_LABELS,
    panel_groups = _BTCV_GROUPS,
    shortcuts    = _BTCV_SHORTCUTS,
    splits_json  = lambda r: r / "btcv" / "splits.json",
    img_path     = lambda r, sid, sp: (
        r / "btcv"
        / ("Testing" if sp == "test" else "Training")
        / "img" / f"img{sid}.nii.gz"
    ),
    lbl_path     = lambda r, sid, sp: (
        None if sp == "test"
        else r / "btcv" / "Training" / "label" / f"label{sid}.nii.gz"
    ),
)

CONFIGS: dict[str, DatasetConfig] = {"oasis1": OASIS1, "btcv": BTCV}


# ── resolve dataset & files ─────────────────────────────────────────────────
SAVE_MODE    = args.save
SPLIT        = args.split
DATASET_ROOT = Path(args.dataset_root)
cfg          = CONFIGS[args.dataset]

with open(cfg.splits_json(DATASET_ROOT)) as f:
    split_ids: list[str] = json.load(f)["splits"][SPLIT]

image_files: list[str]           = []
label_files: list[Optional[str]] = []
for sid in split_ids:
    ip = cfg.img_path(DATASET_ROOT, sid, SPLIT)
    lp = cfg.lbl_path(DATASET_ROOT, sid, SPLIT)
    if ip.exists():
        image_files.append(str(ip))
        label_files.append(str(lp) if (lp is not None and lp.exists()) else None)

print(f"Dataset : {cfg.name}")
print(f"Split   : {SPLIT}  ({len(image_files)} images)")
print(f"Labels  : {sum(1 for l in label_files if l)} annotated")


# ── label colors (golden-angle hue) ────────────────────────────────────────
COLORS: dict[int, np.ndarray] = {}
for _lbl in cfg.label_dict:
    _h = (_lbl * 137.508) % 360 / 360.0
    _s = min(0.65 + (_lbl % 5) * 0.06, 1.0)
    _v = min(0.82 + (_lbl % 3) * 0.05, 1.0)
    COLORS[_lbl] = hsv_to_rgb([_h, _s, _v])

PANEL_ORDER: list[tuple[str, object]] = []
for _g_name, _g_keys in cfg.panel_groups:
    PANEL_ORDER.append(("header", _g_name))
    for _k in _g_keys:
        PANEL_ORDER.append(("label", _k))

N_LABELS  = len(cfg.label_dict)
N_HEADERS = len(cfg.panel_groups)


# ── layout constants ────────────────────────────────────────────────────────
PANEL_FRAC = 0.175
VIEW_LEFT  = PANEL_FRAC + 0.014   # gap between panel and views
VIEW_RIGHT = 0.997
VIEWS_W    = VIEW_RIGHT - VIEW_LEFT
EACH_W     = VIEWS_W / 3.0
TOP        = 0.955
BOTTOM     = 0.225                 # raised to keep slider labels clear of panel

ROW_H = 1.0
HDR_H = 1.3
PANEL_TOTAL_H = N_LABELS * ROW_H + N_HEADERS * HDR_H + 2.0   # 2.0 top/bottom pad

# Sliders — left edge is far enough right that labels don't reach the panel
SL_LEFT_X = VIEW_LEFT + 0.060
SL_W      = VIEW_RIGHT - SL_LEFT_X - 0.008
SL_H      = 0.022
SL_BOT    = 0.020

BTN_X = 0.004
BTN_W = PANEL_FRAC - 0.012
BTN_H = 0.028
GAP   = 0.005

SWATCH_X = 0.018; SWATCH_W = 0.07
CHECK_X  = 0.120; TEXT_X   = 0.180
TEXT_FS  = 5.9


# ── figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(23, 10), facecolor="#0d1117")
if not SAVE_MODE:
    try:
        fig.canvas.manager.set_window_title(
            f"MedSeg Viewer — {cfg.name} [{SPLIT}]"
        )
    except Exception:
        pass

panel_ax = fig.add_axes([0.003, BOTTOM, PANEL_FRAC - 0.006, TOP - BOTTOM])
panel_ax.set_facecolor("#161b22")
panel_ax.set_xticks([]); panel_ax.set_yticks([])
for _sp in panel_ax.spines.values():
    _sp.set_color("#30363d")

ax_sag = fig.add_axes([VIEW_LEFT,            BOTTOM, EACH_W - GAP, TOP - BOTTOM])
ax_cor = fig.add_axes([VIEW_LEFT +   EACH_W, BOTTOM, EACH_W - GAP, TOP - BOTTOM])
ax_axi = fig.add_axes([VIEW_LEFT + 2*EACH_W, BOTTOM, EACH_W - GAP, TOP - BOTTOM])
for _ax in (ax_sag, ax_cor, ax_axi):
    _ax.set_facecolor("#0d1117"); _ax.set_xticks([]); _ax.set_yticks([])

sl_sag_ax = fig.add_axes([SL_LEFT_X, SL_BOT + 0.115, SL_W, SL_H], facecolor="#161b22")
sl_cor_ax = fig.add_axes([SL_LEFT_X, SL_BOT + 0.065, SL_W, SL_H], facecolor="#161b22")
sl_axi_ax = fig.add_axes([SL_LEFT_X, SL_BOT + 0.015, SL_W, SL_H], facecolor="#161b22")

_SLIDER_KW = dict(
    color="#238636", track_color="#21262d",
    handle_style={"facecolor": "#58a6ff", "edgecolor": "#58a6ff", "size": 8},
)
sx = Slider(sl_sag_ax, "Sag", 0, 1, valinit=0, valstep=1, **_SLIDER_KW)
sy = Slider(sl_cor_ax, "Cor", 0, 1, valinit=0, valstep=1, **_SLIDER_KW)
sz = Slider(sl_axi_ax, "Axi", 0, 1, valinit=0, valstep=1, **_SLIDER_KW)
for _sl in (sx, sy, sz):
    _sl.label.set_color("#8b949e"); _sl.valtext.set_color("#58a6ff")

# dataset / split label (non-interactive)
fig.text(
    BTN_X + BTN_W / 2, SL_BOT + 0.115 + SL_H / 2,
    f"{cfg.name}  [{SPLIT}]",
    ha="center", va="center", fontsize=7.5,
    color="#58a6ff", fontweight="bold",
)

if not SAVE_MODE:
    _btn_on_ax  = fig.add_axes([BTN_X, SL_BOT + 0.065, BTN_W, BTN_H],
                               facecolor="#21262d")
    _btn_off_ax = fig.add_axes([BTN_X, SL_BOT + 0.015, BTN_W, BTN_H],
                               facecolor="#21262d")
    btn_on  = Button(_btn_on_ax,  "All ON",  color="#21262d", hovercolor="#30363d")
    btn_off = Button(_btn_off_ax, "All OFF", color="#21262d", hovercolor="#30363d")
    for _b in (btn_on, btn_off):
        _b.label.set_color("#e6edf3"); _b.label.set_fontsize(8)


def _rebuild_slider(host_ax, label, dim, init_val):
    host_ax.clear(); host_ax.set_facecolor("#161b22")
    sl = Slider(host_ax, label, 0, dim - 1,
                valinit=init_val, valstep=1, **_SLIDER_KW)
    sl.label.set_color("#8b949e"); sl.valtext.set_color("#58a6ff")
    return sl


# ── viewer state ────────────────────────────────────────────────────────────
current_case  = 0
active_labels: set[int] = set(cfg.label_dict.keys())
image = label_vol = None
sx_mm = sy_mm = sz_mm = 1.0

# y-centre of each label row in panel data coords — populated by rebuild_panel()
_row_y_map: dict[int, float] = {}


# ── panel ───────────────────────────────────────────────────────────────────
def rebuild_panel():
    _row_y_map.clear()
    panel_ax.clear()
    panel_ax.set_facecolor("#161b22")
    panel_ax.set_xticks([]); panel_ax.set_yticks([])
    panel_ax.set_xlim(0, 1)
    panel_ax.set_ylim(0, PANEL_TOTAL_H)
    for _sp in panel_ax.spines.values():
        _sp.set_color("#30363d")

    y = PANEL_TOTAL_H - 1.0      # top padding
    for kind, val in PANEL_ORDER:
        if kind == "header":
            panel_ax.text(0.5, y, f"── {val} ──",
                          ha="center", va="center", fontsize=5.5,
                          color="#58a6ff", style="italic",
                          transform=panel_ax.transData)
            y -= HDR_H
            continue

        lbl    = int(val)
        active = lbl in active_labels
        col    = COLORS[lbl]
        alpha  = 1.0 if active else 0.18
        cy     = y + 0.05
        _row_y_map[lbl] = cy

        panel_ax.add_patch(mpatches.FancyBboxPatch(
            (SWATCH_X, y - 0.38), SWATCH_W, 0.72,
            boxstyle="round,pad=0.02",
            facecolor=col, edgecolor="none", alpha=alpha,
            transform=panel_ax.transData, clip_on=True,
        ))
        panel_ax.text(CHECK_X, cy, "✓" if active else "·",
                      fontsize=7,
                      color="#58a6ff" if active else "#30363d",
                      va="center", ha="center",
                      transform=panel_ax.transData)
        panel_ax.text(TEXT_X, cy,
                      f"{lbl:>2}  {cfg.label_dict[lbl]}",
                      fontsize=TEXT_FS,
                      color="#dde4ee" if active else "#3d444d",
                      va="center", ha="left",
                      transform=panel_ax.transData)
        y -= ROW_H

    panel_ax.set_title(f"LABELS  {len(active_labels)}/{N_LABELS}",
                       color="#8b949e", fontsize=7, pad=3, loc="left")
    if not SAVE_MODE:
        fig.canvas.draw_idle()


# ── panel click — button_press_event + row hit-test ────────────────────────
# Fixes: original pick_event required clicking exactly on tiny 5.9pt text.
# Now the entire row height is a valid click target.
def on_panel_click(event):
    if event.button != 1 or event.inaxes is not panel_ax:
        return
    y = event.ydata
    if y is None:
        return
    best_lbl, best_d = None, ROW_H * 0.65   # tolerance: 65% of a row height
    for lbl, cy in _row_y_map.items():
        d = abs(y - cy)
        if d < best_d:
            best_d, best_lbl = d, lbl
    if best_lbl is not None:
        active_labels.symmetric_difference_update({best_lbl})
        rebuild_panel()
        update(None)


if not SAVE_MODE:
    fig.canvas.mpl_connect("button_press_event", on_panel_click)


# ── overlay ─────────────────────────────────────────────────────────────────
def make_overlay(seg: np.ndarray) -> np.ndarray:
    h, w = seg.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for lbl in active_labels:
        m = seg == lbl
        if m.any():
            rgba[m, :3] = COLORS[lbl]; rgba[m, 3] = 0.55
    return rgba


# ── crosshairs ───────────────────────────────────────────────────────────────
COL_SAG = "#ff79c6"; COL_COR = "#8be9fd"; COL_AXI = "#f1fa8c"
_LW = 0.9; _ALPHA = 0.85

def _crosshair(ax, h_row, v_col, hc, vc, nr, nc):
    h_row = int(np.clip(h_row, 0, nr - 1))
    v_col = int(np.clip(v_col, 0, nc - 1))
    ax.axhline(h_row, color=hc, lw=_LW, alpha=_ALPHA, ls="--")
    ax.axvline(v_col, color=vc, lw=_LW, alpha=_ALPHA, ls="--")
    ax.plot(v_col, h_row, "+", color="white", ms=9, mew=1.3, alpha=0.9, zorder=10)


# ── update ───────────────────────────────────────────────────────────────────
def update(_val):
    if image is None:
        return
    xi, yi, zi = int(sx.val), int(sy.val), int(sz.val)
    nx, ny, nz = image.shape

    for ax in (ax_sag, ax_cor, ax_axi):
        ax.clear(); ax.set_facecolor("#0d1117")
        ax.set_xticks([]); ax.set_yticks([])

    def _show(ax, img_sl, seg_sl, asp, title, hr, vc, hc, vc2, nr, nc):
        ax.imshow(img_sl, cmap="gray", aspect=asp, interpolation="bilinear")
        ax.imshow(make_overlay(seg_sl), aspect=asp, interpolation="nearest")
        _crosshair(ax, hr, vc, hc, vc2, nr, nc)
        ax.set_title(title, fontsize=8.5, color="#8b949e", pad=3)

    _show(ax_sag,
          np.rot90(image[xi, :, :]), np.rot90(label_vol[xi, :, :]),
          sz_mm / sy_mm, f"Sagittal  x={xi}",
          nz - 1 - zi, yi, COL_AXI, COL_COR, nz, ny)

    _show(ax_cor,
          np.rot90(image[:, yi, :]), np.rot90(label_vol[:, yi, :]),
          sz_mm / sx_mm, f"Coronal   y={yi}",
          nz - 1 - zi, xi, COL_AXI, COL_SAG, nz, nx)

    _show(ax_axi,
          np.rot90(image[:, :, zi]), np.rot90(label_vol[:, :, zi]),
          sy_mm / sx_mm, f"Axial     z={zi}",
          ny - 1 - yi, xi, COL_COR, COL_SAG, ny, nx)

    shown = sorted(active_labels)[:20]
    if shown:
        ax_axi.legend(
            handles=[mpatches.Patch(color=COLORS[l],
                                    label=f"{l:>2} {cfg.label_dict[l][:18]}")
                     for l in shown],
            loc="lower right", fontsize=4.0, framealpha=0.6,
            facecolor="#0d1117", edgecolor="#30363d",
            labelcolor="#dde4ee", ncol=1, handlelength=1.1,
            borderpad=0.4, labelspacing=0.25,
        )

    has_lbl = label_files[current_case] is not None
    fig.suptitle(
        f"MedSeg Viewer  │  {cfg.name}  [{SPLIT.upper()}]  "
        f"Case {current_case + 1}/{len(image_files)}"
        f"  │  {len(active_labels)} labels active"
        + ("  │  no labels" if not has_lbl else ""),
        fontsize=9.5, color="#58a6ff",
        y=0.992, x=VIEW_LEFT + VIEWS_W / 2,
    )
    if not SAVE_MODE:
        fig.canvas.draw_idle()


# ── load case ────────────────────────────────────────────────────────────────
def load_case(idx: int):
    global image, label_vol, sx_mm, sy_mm, sz_mm, sx, sy, sz

    nii = nib.load(image_files[idx])
    image = nii.get_fdata()
    sx_mm, sy_mm, sz_mm = nii.header.get_zooms()[:3]

    if label_files[idx]:
        label_vol = nib.load(label_files[idx]).get_fdata().astype(np.int32)
    else:
        label_vol = np.zeros_like(image, dtype=np.int32)

    nx, ny, nz = image.shape
    sx = _rebuild_slider(sl_sag_ax, f"Sag 0-{nx-1}", nx, nx // 2)
    sy = _rebuild_slider(sl_cor_ax, f"Cor 0-{ny-1}", ny, ny // 2)
    sz = _rebuild_slider(sl_axi_ax, f"Axi 0-{nz-1}", nz, nz // 2)
    if not SAVE_MODE:
        sx.on_changed(update); sy.on_changed(update); sz.on_changed(update)
    update(None)


# ── button callbacks ─────────────────────────────────────────────────────────
if not SAVE_MODE:
    def _cb_all_on(_e):
        active_labels.update(cfg.label_dict.keys())
        rebuild_panel(); update(None)

    def _cb_all_off(_e):
        active_labels.clear()
        rebuild_panel(); update(None)

    btn_on.on_clicked(_cb_all_on)
    btn_off.on_clicked(_cb_all_off)


# ── keyboard ─────────────────────────────────────────────────────────────────
if not SAVE_MODE:
    def on_key(event):
        global current_case

        key = event.key
        if key == "right":
            current_case = (current_case + 1) % len(image_files)
            load_case(current_case)
        elif key == "left":
            current_case = (current_case - 1) % len(image_files)
            load_case(current_case)
        elif key == "a":
            _cb_all_on(None)
        elif key == "n":
            _cb_all_off(None)
        elif key in cfg.shortcuts:
            # select ONLY those labels (focus mode)
            _, lbls = cfg.shortcuts[key]
            active_labels.clear(); active_labels.update(lbls)
            rebuild_panel(); update(None)

    fig.canvas.mpl_connect("key_press_event", on_key)


# ── scroll zoom ───────────────────────────────────────────────────────────────
if not SAVE_MODE:
    def on_scroll(event):
        ax = event.inaxes
        if ax not in (ax_sag, ax_cor, ax_axi):
            return
        f  = 0.85 if event.button == "up" else 1.15
        cx = np.mean(ax.get_xlim()); cy = np.mean(ax.get_ylim())
        rx = (ax.get_xlim()[1] - ax.get_xlim()[0]) * f / 2
        ry = (ax.get_ylim()[1] - ax.get_ylim()[0]) * f / 2
        ax.set_xlim(cx - rx, cx + rx); ax.set_ylim(cy - ry, cy + ry)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)


# ── entry point ───────────────────────────────────────────────────────────────
rebuild_panel()

if SAVE_MODE:
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    idx  = min(args.case, len(image_files) - 1)
    stem = Path(image_files[idx]).name
    print(f"Rendering case {idx} ({stem}) ...", flush=True)
    load_case(idx)
    out = save_dir / f"{args.dataset}_{SPLIT}_{stem}_case{idx:04d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"Saved → {out}")

else:
    print("Loading first case ...", flush=True)
    load_case(current_case)
    print(f"\n── MedSeg Viewer  {cfg.name}  [{SPLIT.upper()}]"
          f"  {len(image_files)} cases ──")
    print("  ← / →    : switch case")
    print("  A / N    : all labels ON / OFF")
    print("  Click    : toggle label in panel")
    print("  Scroll   : zoom in/out")
    if cfg.shortcuts:
        for k, (desc, _) in cfg.shortcuts.items():
            print(f"  {k.upper():8s} : {desc}")
    print()
    plt.show()
