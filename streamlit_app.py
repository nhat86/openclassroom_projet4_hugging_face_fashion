"""
Fashion Trend Intelligence - Application de présentation du projet
Segmentation sémantique de vêtements via l'API Hugging Face (SegFormer B3 Clothes).

Lancement : streamlit run streamlit_app.py
"""

import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from seg_utils import (
    API_URL, CLASS_MAPPING, DATASET_DIR, ID_TO_LABEL, PRED_DIR, RESULTS_DIR,
    classes_present, colorize_mask, compute_iou_per_class, get_api_token,
    get_ground_truth_mask, get_predicted_mask, list_image_ids,
)

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fashion Trend Intelligence",
    page_icon="👗",
    layout="wide",
)

st.sidebar.title("Fashion Trend Intelligence")
st.sidebar.caption("Segmentation sémantique de vêtements — Nhat VO")
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Pipeline", "Démo segmentation", "Performances (IoU)",
     "Passage à l'échelle"],
)

st.sidebar.divider()
if DATASET_DIR is None:
    st.sidebar.error("Dataset introuvable (dossier IMG/ + Mask/)")
else:
    st.sidebar.success(f"{len(list_image_ids())} images détectées")

api_token = get_api_token()
if not api_token:
    api_token = st.sidebar.text_input(
        "Token Hugging Face", type="password",
        help="Requis uniquement pour lancer de nouvelles segmentations.")
else:
    st.sidebar.info("Token API chargé depuis .env")


# ---------------------------------------------------------------------------
# Page : Accueil
# ---------------------------------------------------------------------------

if page == "Accueil":
    st.title("Fashion Trend Intelligence")
    st.subheader("Segmentation sémantique de vêtements par IA")
    st.caption("Analyse de tendances sur images d'influenceurs mode 2024 — "
               "Formation Développeur IA, OpenClassrooms")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Objectif")
        st.markdown(
            "Automatiser l'analyse des tenues portées par les influenceurs mode "
            "afin d'identifier les tendances émergentes et d'alimenter les "
            "décisions éditoriales et commerciales de **ModeTrend**."
        )
    with c2:
        st.markdown("### Approche technique")
        st.markdown(
            "Utilisation du modèle **SegFormer B3 Clothes** via l'API "
            "d'inférence Hugging Face pour segmenter chaque image en "
            "**18 catégories** : vêtements, accessoires, chaussures, etc."
        )

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images du dataset", len(list_image_ids()) or "50")
    m2.metric("Classes de segmentation", len(CLASS_MAPPING))
    m3.metric("Modèle", "SegFormer B3")
    m4.metric("IoU moyen constaté", "0,6 – 0,85")

    st.markdown("**Stack :** Python · Hugging Face Inference API · Pillow · "
                "NumPy · Pandas · Matplotlib · Streamlit")


# ---------------------------------------------------------------------------
# Page : Pipeline
# ---------------------------------------------------------------------------

elif page == "Pipeline":
    st.title("Fonctionnement du pipeline")

    steps = [
        ("1. Chargement des images",
         "Lecture des images depuis le dossier local `IMG/` "
         "(50 photos d'influenceurs mode 2024)."),
        ("2. Appel à l'API Hugging Face",
         "Envoi de chaque image en requête HTTP POST vers "
         "`sayeed99/segformer_b3_clothes`, avec authentification par token."),
        ("3. Décodage des masques",
         "L'API renvoie un masque par classe détectée, encodé en base64. "
         "Chaque masque est converti en tableau NumPy puis redimensionné "
         "à la taille de l'image originale."),
        ("4. Fusion des masques",
         "Les masques par classe sont combinés en une carte de segmentation "
         "unique : chaque pixel reçoit l'identifiant de sa classe (0 à 17)."),
        ("5. Évaluation et export",
         "Comparaison avec le masque de référence (ground truth) par le "
         "calcul de l'IoU (Intersection over Union), par classe et par image."),
    ]
    for title, body in steps:
        with st.expander(title, expanded=True):
            st.markdown(body)

    st.info(
        "Modèle utilisé : "
        "[sayeed99/segformer_b3_clothes]"
        "(https://huggingface.co/sayeed99/segformer_b3_clothes)",
        icon="🤗",
    )
    st.code(API_URL, language="text")


# ---------------------------------------------------------------------------
# Page : Démo segmentation
# ---------------------------------------------------------------------------

elif page == "Démo segmentation":
    st.title("Démo — Segmentation d'une image")

    if DATASET_DIR is None:
        st.error("Dataset introuvable.")
        st.stop()

    ids = list_image_ids()
    image_id = st.selectbox("Choisir une image", ids,
                            format_func=lambda i: f"image_{i}.png")

    image_path = DATASET_DIR / "IMG" / f"image_{image_id}.png"
    gt_path = DATASET_DIR / "Mask" / f"mask_{image_id}.png"
    original = Image.open(image_path)

    cached = (PRED_DIR / f"pred_{image_id}.png").exists()
    if cached:
        st.caption("Masque prédit chargé depuis le cache local.")
    elif not api_token:
        st.warning("Pas de token API et aucun masque en cache pour cette image.")
    else:
        st.caption("Masque non calculé — lancement via l'API ci-dessous.")

    run = cached or st.button("Lancer la segmentation", type="primary")

    pred_mask = None
    if run:
        with st.spinner("Segmentation en cours..."):
            try:
                pred_mask, _ = get_predicted_mask(image_id, api_token)
            except Exception as e:
                st.error(f"Erreur API : {e}")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Image originale**")
        st.image(original, use_container_width=True)
    with cols[1]:
        st.markdown("**Masque prédit (modèle)**")
        if pred_mask is not None:
            st.image(colorize_mask(pred_mask), use_container_width=True)
            st.caption("Classes : " + ", ".join(classes_present(pred_mask)))
        else:
            st.info("En attente de segmentation")
    with cols[2]:
        st.markdown("**Masque réel (ground truth)**")
        if gt_path.exists():
            gt_mask = get_ground_truth_mask(image_id)
            st.image(colorize_mask(gt_mask), use_container_width=True)
            st.caption("Classes : " + ", ".join(classes_present(gt_mask)))
        else:
            st.info("Pas de masque de référence")

    if pred_mask is not None and gt_path.exists():
        gt_mask = get_ground_truth_mask(image_id, original.size)
        ious = compute_iou_per_class(pred_mask, gt_mask)
        if ious:
            st.divider()
            mean_iou = np.mean([i / u for i, u in ious.values()])
            st.metric("IoU moyen sur cette image", f"{mean_iou:.3f}")
            df = pd.DataFrame(
                [(ID_TO_LABEL[c], i / u) for c, (i, u) in ious.items()],
                columns=["Classe", "IoU"],
            ).sort_values("IoU", ascending=False)
            fig = px.bar(df, x="Classe", y="IoU", range_y=[0, 1],
                         title=f"IoU par classe — image_{image_id}.png")
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page : Performances (IoU)
# ---------------------------------------------------------------------------

elif page == "Performances (IoU)":
    st.title("Performances du modèle sur le dataset")

    if DATASET_DIR is None:
        st.error("Dataset introuvable.")
        st.stop()

    csv_img = RESULTS_DIR / "iou_per_image.csv"
    csv_cls = RESULTS_DIR / "iou_per_class.csv"

    df_img = pd.read_csv(csv_img) if csv_img.exists() else None
    df_cls = pd.read_csv(csv_cls) if csv_cls.exists() else None

    ids = list_image_ids()
    n_missing = sum(1 for i in ids if not (PRED_DIR / f"pred_{i}.png").exists())
    st.write(
        f"**{len(ids)}** images · **{len(ids) - n_missing}** masques prédits "
        f"en cache · **{n_missing}** à calculer via l'API"
    )

    if n_missing and not api_token:
        st.warning("Token API requis pour calculer les masques manquants.")

    if st.button("Calculer / compléter les IoU sur tout le dataset",
                 disabled=bool(n_missing and not api_token)):
        rows_img, agg = [], {}
        progress = st.progress(0.0, text="Traitement des images...")
        for k, i in enumerate(ids):
            progress.progress((k + 1) / len(ids),
                              text=f"image_{i}.png ({k + 1}/{len(ids)})")
            try:
                pred, from_cache = get_predicted_mask(i, api_token)
                if not from_cache:
                    time.sleep(0.4)
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
                rows_img.append({"image": f"image_{i}.png",
                                 "mean_iou": np.nan, "n_classes": 0})
                st.warning(f"image_{i}.png : {e}")
        progress.empty()

        df_img = pd.DataFrame(rows_img)
        df_cls = pd.DataFrame(
            [(ID_TO_LABEL[c], a[0] / a[1]) for c, a in agg.items()],
            columns=["Classe", "IoU"],
        ).sort_values("IoU", ascending=False)
        df_img.to_csv(csv_img, index=False)
        df_cls.to_csv(csv_cls, index=False)

    if df_img is not None and df_cls is not None:
        valid = df_img["mean_iou"].dropna()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("IoU moyen global", f"{valid.mean():.3f}")
        m2.metric("IoU médian", f"{valid.median():.3f}")
        m3.metric("Min", f"{valid.min():.3f}")
        m4.metric("Max", f"{valid.max():.3f}")

        st.divider()
        st.subheader("IoU moyen par classe (poolé sur tout le dataset)")
        fig = px.bar(df_cls, x="Classe", y="IoU", range_y=[0, 1],
                     color="IoU", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribution de l'IoU moyen par image")
        fig2 = px.histogram(df_img.dropna(), x="mean_iou", nbins=20,
                            range_x=[0, 1],
                            labels={"mean_iou": "IoU moyen"})
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Données par image"):
            st.dataframe(df_img.sort_values("mean_iou"),
                         use_container_width=True)
    else:
        st.info("Aucun résultat calculé pour l'instant. "
                "Lance le calcul ci-dessus (les masques déjà en cache "
                "seront réutilisés).")


# ---------------------------------------------------------------------------
# Page : Passage à l'échelle
# ---------------------------------------------------------------------------

elif page == "Passage à l'échelle":
    st.title("Passage à l'échelle et coût d'utilisation")

    st.markdown(
        "Solution envisagée : **Hugging Face Inference Endpoint** dédié "
        "(GPU T4). Ajuste les paramètres pour estimer le budget."
    )

    c1, c2, c3 = st.columns(3)
    volume = c1.number_input("Images / mois", value=500_000, step=50_000)
    tps = c2.number_input("Temps par image (s)", value=0.5, step=0.1)
    prix = c3.number_input("Prix GPU T4 (€/h)", value=0.5, step=0.1)

    heures_mois = 720
    besoin_h = volume * tps / 3600
    instances = max(1, int(np.ceil(besoin_h / heures_mois)))
    cout = instances * prix * heures_mois
    cout_image = cout / volume if volume else 0

    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Charge de calcul", f"{besoin_h:,.0f} h/mois")
    r2.metric("Instances GPU T4", instances)
    r3.metric("Budget estimé", f"{cout:,.0f} €/mois")
    r4.metric("Coût par image", f"{cout_image:.5f} €")

    st.caption(
        "Hypothèse : instance active 24/7 (720 h/mois). "
        "Exemple de la présentation : 500 000 images/mois à ~0,5 s/image "
        "→ 1 instance → **360 €/mois** (~0,0007 €/image)."
    )
