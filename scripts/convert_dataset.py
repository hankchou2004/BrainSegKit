"""
將 OASIS-1 資料轉換為 FreeSurfer .nii.gz 資料集（41 類 aseg）。

    T1.mgz   →  dataset/freesurfer/imagesTr/{subject}_0000.nii.gz
    aseg.mgz →  dataset/freesurfer/labelsTr/{subject}.nii.gz

輸出目錄：
    dataset/freesurfer/
        imagesTr/  labelsTr/  imagesTs/  labelsTs/

執行：
    python convert_dataset.py [--no-delete]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


BASE       = Path('/home/hank/medical_segmention')
FS_ROOT    = BASE / 'brain_data' / 'oasis1_freesurfer'
SPLITS_JSON = BASE / 'dataset' / 'oasis1_freesurfer' / 'splits.json'
OUT_ROOT   = BASE / 'dataset' / 'oasis1_freesurfer'


# ── 工具函式 ────────────────────────────────────────────────────────────────

def mgz_to_nii(src: Path, dst: Path, dtype):
    img  = nib.load(str(src))
    data = np.asarray(img.dataobj, dtype=dtype)
    nib.save(nib.Nifti1Image(data, img.affine), str(dst))


def out_dirs(split: str) -> tuple[Path, Path]:
    """回傳 (image_dir, label_dir)"""
    if split in ('train', 'val'):
        return OUT_ROOT / 'imagesTr', OUT_ROOT / 'labelsTr'
    return OUT_ROOT / 'imagesTs', OUT_ROOT / 'labelsTs'


# ── 主程式 ──────────────────────────────────────────────────────────────────

def convert_freesurfer(splits: dict[str, str]) -> list[str]:
    print('\n── FreeSurfer 轉換中 ──────────────────────────────')
    errors = []
    total  = len(splits)

    for i, (sid, split) in enumerate(splits.items(), 1):
        t1_src   = FS_ROOT / sid / 'mri' / 'T1.mgz'
        aseg_src = FS_ROOT / sid / 'mri' / 'aseg.mgz'

        if not t1_src.exists() or not aseg_src.exists():
            errors.append(f'MISSING: {sid}')
            print(f'[{i:3d}/{total}] SKIP  {sid}')
            continue

        img_dir, lbl_dir = out_dirs(split)
        try:
            mgz_to_nii(t1_src,   img_dir / f'{sid}_0000.nii.gz', np.float32)
            mgz_to_nii(aseg_src, lbl_dir / f'{sid}.nii.gz',       np.int16)
            print(f'[{i:3d}/{total}] OK  {sid}  ({split})')
        except Exception as e:
            errors.append(f'{sid}: {e}')
            print(f'[{i:3d}/{total}] ERR {sid}: {e}')

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-delete', action='store_true',
                        help='轉換完成後不刪除 brain_data/')
    args = parser.parse_args()

    # 讀取 splits
    splits: dict[str, str] = {}
    with open(SPLITS_JSON) as f:
        data = json.load(f)
    for split_name, ids in data["splits"].items():
        for sid in ids:
            splits[sid] = split_name
    n_tr = sum(1 for s in splits.values() if s == "train")
    n_va = sum(1 for s in splits.values() if s == "val")
    n_te = sum(1 for s in splits.values() if s == "test")
    print(f'共 {len(splits)} 筆 ({n_tr} train / {n_va} val / {n_te} test)')

    # 建立輸出目錄
    for d in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        (OUT_ROOT / d).mkdir(parents=True, exist_ok=True)

    errors = convert_freesurfer(splits)

    # 摘要
    print('\n' + '=' * 55)
    for sub in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        n = len(list((OUT_ROOT / sub).glob('*.nii.gz')))
        print(f'  {sub:12s} {n:4d} 個檔案')

    if errors:
        print(f'\n錯誤 {len(errors)} 筆：')
        for e in errors:
            print(' ', e)
        print('\n→ 有錯誤，保留 brain_data/，請確認後手動刪除。')
        sys.exit(1)

    if not args.no_delete:
        print('\n刪除原始 brain_data/ ...')
        shutil.rmtree(BASE / 'brain_data')
        print('刪除完成。')

    print(f'\n✓ 完成！資料集位置：{OUT_ROOT}')


if __name__ == '__main__':
    main()
