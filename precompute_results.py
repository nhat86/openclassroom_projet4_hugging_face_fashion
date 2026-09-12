"""Pré-calcule les masques prédits (cache) et les IoU pour tout le dataset.

Usage : python precompute_results.py
Nécessite API_KEY dans le .env du dataset (ou variable d'environnement).
"""

import time

import numpy as np
import pandas as pd
from PIL import Image

from seg_utils import (
    ID_TO_LABEL, PRED_DIR, RESULTS_DIR, compute_iou_per_class, get_api_token,
    get_ground_truth_mask, get_predicted_mask, list_image_ids,
)


def main():
    token = get_api_token()
    ids = list_image_ids()
    if not ids:
        print("Dataset introuvable.")
        return
    if not token:
        print("Pas de token API : seuls les masques déjà en cache seront utilisés.")

    rows_img, agg = [], {}
    for k, i in enumerate(ids):
        try:
            pred, from_cache = get_predicted_mask(i, token)
            if pred is None:
                print(f"[{k + 1}/{len(ids)}] image_{i}: pas de masque (token manquant)")
                continue
            if not from_cache:
                print(f"[{k + 1}/{len(ids)}] image_{i}: segmentée via API")
                time.sleep(0.4)
            else:
                print(f"[{k + 1}/{len(ids)}] image_{i}: cache")
            gt = get_ground_truth_mask(i)
            if (gt.shape[1], gt.shape[0]) != (pred.shape[1], pred.shape[0]):
                gt = np.array(Image.fromarray(gt).resize(
                    (pred.shape[1], pred.shape[0]), Image.NEAREST))
            ious = compute_iou_per_class(pred, gt)
            vals = [inter / union for inter, union in ious.values()]
            rows_img.append({
                "image": f"image_{i}.png",
                "mean_iou": float(np.mean(vals)) if vals else np.nan,
                "n_classes": len(ious),
            })
            for cid, (inter, union) in ious.items():
                a = agg.setdefault(cid, [0, 0])
                a[0] += inter
                a[1] += union
        except Exception as e:
            print(f"[{k + 1}/{len(ids)}] image_{i}: ERREUR {e}")
            rows_img.append({"image": f"image_{i}.png",
                             "mean_iou": np.nan, "n_classes": 0})

    df_img = pd.DataFrame(rows_img)
    df_cls = pd.DataFrame(
        [(ID_TO_LABEL[c], a[0] / a[1]) for c, a in agg.items()],
        columns=["Classe", "IoU"],
    ).sort_values("IoU", ascending=False)

    df_img.to_csv(RESULTS_DIR / "iou_per_image.csv", index=False)
    df_cls.to_csv(RESULTS_DIR / "iou_per_class.csv", index=False)

    print("\n=== Résultats ===")
    print(f"IoU moyen global : {df_img['mean_iou'].mean():.3f}")
    print(f"IoU médian       : {df_img['mean_iou'].median():.3f}")
    print(df_cls.to_string(index=False))


if __name__ == "__main__":
    main()
