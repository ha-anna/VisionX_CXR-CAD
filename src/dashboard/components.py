import binascii
import html
import os

import streamlit as st
from i18n import disease_name, t
from image_utils import read_heatmap
from PIL import Image
from styles import caption, compact_section, notice


def score_chart(scores, detections=()):
    rows = []
    for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        label = html.escape(disease_name(name))
        rows.append(
            f'<div class="vx-score"><span>{label}{" ●" if name in detections else ""}</span>'
            f'<div class="vx-track"><div class="vx-fill" '
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

    heatmap = None

    if result and result.get("gradcam_heatmap"):
        try:
            heatmap = read_heatmap(result["gradcam_heatmap"])
            if not preview and os.getenv("GRADCAM_IS_OVERLAY", "true").lower() != "true":
                heatmap = Image.blend(image, heatmap.resize(image.size), alpha=0.4)
        except (ValueError, OSError, binascii.Error, Image.DecompressionBombError):
            notice(
                "warning",
                t(
                    "The explanation image could not be displayed. "
                    "Prediction scores are still available."
                ),
            )

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

    if heatmap is not None and selection == "Compare":
        original_col, explanation_col = st.columns(2)
        with original_col:
            st.image(image, caption=t("Original image"), width="stretch")
        with explanation_col:
            st.image(
                heatmap,
                caption=(
                    t("Illustrative overlay · Not model-generated")
                    if preview
                    else f"Grad-CAM · {disease_name(result['top1_disease'])}"
                ),
                width="stretch",
            )
    elif selection == "Explanation" and heatmap is not None:
        st.image(
            heatmap,
            caption=t("Illustrative overlay") if preview else t("Grad-CAM explanation"),
            width="stretch",
        )
    else:
        st.image(image, caption=t("Original image"), width="stretch")

    with compact_section("explanation"):
        if result and heatmap is None:
            caption(t("An explanation image is not available for this result."))

        if heatmap is not None:
            with st.expander(t("How to read the explanation")):
                if preview:
                    st.write(t("This colored overlay illustrates the layout. It is not Grad-CAM."))
                else:
                    st.write(
                        t(
                            "Grad-CAM highlights regions influencing the selected disease score. "
                            "It does not verify disease location."
                        )
                    )


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
        else:
            st.write(
                t(
                    "Detection thresholds are not configured. "
                    "Model scores and the Grad-CAM overlay are shown below."
                )
            )
            caption(t("This does not establish that the image is clinically normal."))
        st.space(1)

    with st.expander(t("Model details")):
        st.write(t("DenseNet-121 · 14 independent disease outputs"))

        if preview:
            st.write(t("Illustrative scores; no model was called."))
        else:
            st.write(t("Model version:"), result.get("model_version", t("Not supplied by API")))
            st.write(
                t("Processing time (ms):"),
                result.get("inference_time_ms", t("Not supplied by API")),
            )


def render_scores(result):
    st.subheader(t("All disease scores"), anchor=False)
    score_chart(result["predictions"], result["detected_diseases"])
    caption(
        t(
            "Scores range from 0 to 1; they are not disease probabilities. "
            "A dot marks a detected finding. "
            "Detection uses disease-specific thresholds from the API."
        )
    )
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
            render_scores(result)
    else:
        render_detections(result, preview)
        render_scores(result)
