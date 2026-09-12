"""Logique métier partagée : API Hugging Face, masques, IoU."""

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR / "results"
PRED_DIR = RESULTS_DIR / "pred_masks"
PRED_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"

CLASS_MAPPING = {
    "Background": 0, "Hat": 1, "Hair": 2, "Sunglasses": 3,
    "Upper-clothes": 4, "Skirt": 5, "Pants": 6, "Dress": 7,
    "Belt": 8, "Left-shoe": 9, "Right-shoe": 10, "Face": 11,
    "Left-leg": 12, "Right-leg": 13, "Left-arm": 14, "Right-arm": 15,
    "Bag": 16, "Scarf": 17,
}
ID_TO_LABEL = {v: k for k, v in CLASS_MAPPING.items()}

PALETTE = np.array([
    [40, 40, 40],     # 0  Background
    [255, 60, 60],    # 1  Hat
    [139, 69, 19],    # 2  Hair
    [0, 220, 220],    # 3  Sunglasses
    [30, 130, 255],   # 4  Upper-clothes
    [220, 0, 220],    # 5  Skirt
    [0, 180, 80],     # 6  Pants
    [255, 140, 0],    # 7  Dress
    [255, 230, 0],    # 8  Belt
    [140, 0, 255],    # 9  Left-shoe
    [190, 110, 255],  # 10 Right-shoe
    [255, 205, 170],  # 11 Face
    [0, 100, 0],      # 12 Left-leg
    [80, 160, 60],    # 13 Right-leg
    [255, 110, 110],  # 14 Left-arm
    [200, 60, 60],    # 15 Right-arm
    [90, 90, 255],    # 16 Bag
    [255, 190, 60],   # 17 Scarf
], dtype=np.uint8)


def find_dataset_dir():
    """Localise le dossier contenant IMG/ et Mask/ (dataset top_influenceurs_2024)."""
    candidates = [
        APP_DIR / "asset" / "top_influenceurs_2024",
        APP_DIR.parent / "top_influenceurs_2024-20250314T143221Z-001+(1)" / "top_influenceurs_2024",
    ]
    if APP_DIR.parent.exists():
        for p in sorted(APP_DIR.parent.iterdir()):
            if p.is_dir():
                candidates.append(p / "top_influenceurs_2024")
                candidates.append(p)
    for c in candidates:
        if (c / "IMG").is_dir() and (c / "Mask").is_dir():
            return c
    return None


DATASET_DIR = find_dataset_dir()

if DATASET_DIR is not None:
    load_dotenv(DATASET_DIR / ".env")
load_dotenv(APP_DIR / ".env")


def list_image_ids():
    if DATASET_DIR is None:
        return []
    ids = []
    for f in (DATASET_DIR / "IMG").glob("image_*.png"):
        try:
            ids.append(int(f.stem.split("_")[1]))
        except ValueError:
            pass
    return sorted(ids)


def get_api_token():
    return os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Segmentation : API Hugging Face + cache disque
# ---------------------------------------------------------------------------

def decode_base64_mask(base64_string, width, height):
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)


def create_masks(results, width, height):
    combined = np.zeros((height, width), dtype=np.uint8)
    for result in results:
        class_id = CLASS_MAPPING.get(result["label"], 0)
        if class_id == 0:
            continue
        mask_array = decode_base64_mask(result["mask"], width, height)
        combined[mask_array > 0] = class_id
    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            combined[mask_array > 0] = 0
    return combined


def segment_image_api(image_path, token, retries=3):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "image/png"}
    data = Path(image_path).read_bytes()
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, data=data, timeout=120)
            if r.status_code == 503 and attempt < retries - 1:
                time.sleep(5)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(3)
    return None


def get_predicted_mask(image_id, token=None):
    """Retourne (masque, depuis_cache). Utilise le cache disque, sinon l'API."""
    cache_path = PRED_DIR / f"pred_{image_id}.png"
    if cache_path.exists():
        return np.array(Image.open(cache_path)), True
    if not token:
        return None, False
    image_path = DATASET_DIR / "IMG" / f"image_{image_id}.png"
    results = segment_image_api(image_path, token)
    width, height = Image.open(image_path).size
    mask = create_masks(results, width, height).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(cache_path)
    return mask, False


def get_ground_truth_mask(image_id, size=None):
    path = DATASET_DIR / "Mask" / f"mask_{image_id}.png"
    gt = np.array(Image.open(path))
    if size is not None and (gt.shape[1], gt.shape[0]) != size:
        gt = np.array(Image.fromarray(gt).resize(size, Image.NEAREST))
    return gt


def colorize_mask(mask):
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]


def classes_present(mask):
    return [ID_TO_LABEL[c] for c in np.unique(mask) if c in ID_TO_LABEL]


def compute_iou_per_class(pred, gt):
    """Retourne {class_id: (intersection, union)} pour les classes présentes."""
    ious = {}
    for cid in range(len(CLASS_MAPPING)):
        p, g = pred == cid, gt == cid
        union = int((p | g).sum())
        if union:
            ious[cid] = (int((p & g).sum()), union)
    return ious
