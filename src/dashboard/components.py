import binascii
import html
import os

import streamlit as st
from i18n import disease_name, t
from image_utils import read_heatmap
from PIL import Image
from styles import caption, compact_section


def score_chart(scores, detections=()):
    rows = []
    for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        label = html.escape(disease_name(name))
        if score >= 0.50:
            band = "high"
        elif score >= 0.30:
            band = "middle"
        else:
            band = "low"

        rows.append(
            f'<div class="vx-score"><span>{label}{" ●" if name in detections else ""}</span>'
            f'<div class="vx-track"><div class="vx-fill vx-fill--{band}" '
            f'style="width:{score * 100:.3f}%"></div></div>'
            f'<span class="vx-value">{score:.2f}</span></div>'
        )

    st.markdown(
        '<div class="vx-axis-row"><span></span><div class="vx-axis"><span>0.00</span>'
        "<span>0.50</span><span>1.00</span></div><span></span></div>" + "".join(rows),
        unsafe_allow_html=True,
    )


def render_images(image, result, image_focus=False, preview=False):
    if image is None:
        return

    disease = None
    if result and not preview:
        selected_label = result.get("gradcam_disease") or result["top1_disease"]
        disease = disease_name(selected_label)

    heatmap = None
    if result and result.get("gradcam_heatmap"):
        try:
            heatmap = read_heatmap(result["gradcam_heatmap"])
            if not preview and os.getenv("GRADCAM_IS_OVERLAY", "true").lower() != "true":
                heatmap = Image.blend(image, heatmap.resize(image.size), alpha=0.4)
        except (ValueError, OSError, binascii.Error, Image.DecompressionBombError):
            heatmap = None

    selection = "Compare"
    with compact_section("viewer-controls"):
        st.subheader(t("Image viewer"), anchor=False)
        if image_focus and heatmap is not None:
            selection = st.radio(
                t("Display"),
                ["Compare", "Original", "Explanation"],
                format_func=t,
                horizontal=True,
                key="image_display",
            )

    overlay_caption = (
        t("Illustrative overlay · Not model-generated") if preview else f"Grad-CAM · {disease}"
    )

    if result and selection == "Compare":
        original_col, explanation_col = st.columns(2)

        with original_col:
            st.image(image, caption=t("Original image"), width="stretch")

        with explanation_col:
            if heatmap is not None:
                st.image(heatmap, caption=overlay_caption, width="stretch")
            else:
                message = (
                    t("An explanation image is not available for this result.")
                    if preview
                    else t("Grad-CAM is unavailable for {disease}.", disease=disease)
                )
                st.markdown(
                    f'<div class="vx-gradcam-notice" role="status">{html.escape(message)}</div>',
                    unsafe_allow_html=True,
                )

    elif heatmap is not None and selection == "Explanation":
        st.image(heatmap, caption=overlay_caption, width="stretch")

    else:
        st.image(image, caption=t("Original image"), width="stretch")

    with compact_section("explanation"):
        if heatmap is not None and selection != "Original":
            if preview:
                caption(t("This colored overlay illustrates the layout. It is not Grad-CAM."))
            else:
                st.markdown(
                    f"""
                    <div class="vx-cam-legend">
                        <span>{t("Weaker contribution")}</span>
                        <span class="vx-cam-scale" aria-hidden="true"></span>
                        <span>{t("Stronger contribution")}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                caption(
                    t(
                        "Highlighted areas contributed to the DenseNet-121 score for the "
                        "displayed disease. The overlay cannot confirm a diagnosis or "
                        "precisely outline a lesion."
                    )
                )
                st.space(1)


def render_detections(result, preview=False):
    with st.container(key="findings-summary", border=False):
        st.subheader(t("Example findings") if preview else t("Findings summary"), anchor=False)
        caption(t("Scores above their configured disease thresholds."))
        st.space(1)
        diseases = result["detected_diseases"]

        if diseases:
            chips = "".join(
                f'<span class="vx-chip">{html.escape(disease_name(name))}</span>'
                for name in diseases
            )
            st.markdown(f'<div class="vx-chips">{chips}</div>', unsafe_allow_html=True)
        elif result.get("thresholds_applied"):
            st.write(t("No finding met its detection threshold."))
        else:
            st.write(
                t(
                    "Detection thresholds are not configured. "
                    "Model scores and the Grad-CAM overlay are shown below."
                )
            )
            caption(t("This does not establish that the image is clinically normal."))

    with st.expander(t("Model details")):
        st.write(t("DenseNet-121 + EfficientNet-B0 ensemble · 14 independent disease outputs"))
        st.caption(t("Grad-CAM source: DenseNet-121, fold 0"))

        if preview:
            st.write(t("Illustrative scores; no model was called."))
        else:
            st.write(t("Model version:"), result.get("model_version", t("Not supplied by API")))
            st.write(
                t("Processing time (ms):"),
                result.get("inference_time_ms", t("Not supplied by API")),
            )


def render_scores(result, preview=False):
    st.subheader(t("All disease scores"), anchor=False)
    if not preview:
        scores = result["predictions"]
        st.selectbox(
            t("Disease to show in Grad-CAM"),
            options=sorted(scores, key=scores.get, reverse=True),
            format_func=disease_name,
            key="gradcam_disease",
        )
    score_chart(result["predictions"], result["detected_diseases"])
    bands = (
        ("low", t("Green · Below 0.30")),
        ("middle", t("Yellow · 0.30 to below 0.50")),
        ("high", t("Red · 0.50 and above")),
    )
    legend_items = "".join(
        '<span class="vx-score-legend-item">'
        f'<span class="vx-score-legend-dot vx-fill--{band}" aria-hidden="true"></span>'
        f"{html.escape(label)}</span>"
        for band, label in bands
    )
    st.markdown(
        f'<div class="vx-score-legend">{legend_items}</div>',
        unsafe_allow_html=True,
    )
    caption(t("A dot marks a finding that met its own detection threshold."))
    st.space(4)


def render_results(result, preview=False, image_focus=False):
    st.subheader(t("Analysis results"), anchor=False)

    if result and preview:
        caption(t("Illustrative preview · Not model predictions"))
    elif result:
        caption(t("Analysis complete · Review findings and scores below."))
    if not result:
        caption(t("Your results will appear here after you select Analyze image."))
        st.space(4)
        return
    if image_focus:
        detections, scores = st.columns([1, 1.6], gap="large")
        with detections:
            render_detections(result, preview)
        with scores:
            render_scores(result, preview=preview)
    else:
        render_detections(result, preview)
        render_scores(result, preview=preview)
