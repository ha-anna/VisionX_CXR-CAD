import hashlib

import requests
import streamlit as st
from api_client import prediction_request
from components import render_images, render_results
from i18n import t
from image_utils import decode_image
from PIL import Image, UnidentifiedImageError
from preview import SAMPLE_SOURCE, preview_images, preview_response
from settings import MAX_BYTES
from styles import apply_styles, compact_section


def toggle_language():
    st.session_state["ui_language"] = (
        "ko" if st.session_state.get("ui_language", "en") == "en" else "en"
    )


def main():
    st.set_page_config(page_title="VisionX's CXR-CAD System", page_icon="🩻", layout="wide")
    st.session_state.setdefault("ui_language", "en")
    apply_styles(True)

    with compact_section("header"):
        title_col, language_col = st.columns([5, 1])
        with title_col:
            st.markdown(
                '<h1 class="vx-title"><a href="https://github.com/ha-anna/VisionX_CXR-CAD" '
                'target="_blank" rel="noopener noreferrer" '
                'aria-label="VisionX GitHub repository">VisionX\'s CXR-CAD System↗</a></h1>',
                unsafe_allow_html=True,
            )
        with language_col:
            english = st.session_state["ui_language"] == "en"
            st.button(
                "한국어" if english else "English",
                key="language_toggle",
                help="한국어로 전환" if english else "Switch to English",
                on_click=toggle_language,
            )
        st.caption(t("Upload a chest X-ray, analyze it, and review the model's findings."))
        image_focus = st.checkbox(
            t("Enlarge image viewer"),
            value=False,
            key="image_focus",
            help=t("Give images more space and place results below them."),
        )
        with st.expander(t("Demo options")):
            preview = st.toggle(t("Show illustrative results"), value=False, key="preview_toggle")
            st.caption(t("Preview the interface without sending an image for analysis."))
        if preview:
            st.warning(
                t("Preview mode · Scores are illustrative and are not predictions for your image.")
            )
        st.divider()

    apply_styles(image_focus)
    left, right = (
        (st.container(), st.container()) if image_focus else st.columns([1, 1.05], gap="large")
    )

    with left:
        with compact_section("upload"):
            st.subheader(t("Upload image"), anchor=False)
            uploaded = st.file_uploader(
                "Upload a chest X-ray / 흉부 X선 영상 업로드",
                type=["png", "jpg", "jpeg"],
                key="xray_upload",
                label_visibility="collapsed",
                max_upload_size=10,
            )
            raw = uploaded.getvalue() if uploaded else None
            identity = hashlib.sha256(raw).hexdigest() if raw else None
            if st.session_state.get("image_identity") != identity:
                st.session_state["image_identity"] = identity
                st.session_state.pop("result", None)
            if st.session_state.get("preview_mode") != preview:
                st.session_state["preview_mode"] = preview
                st.session_state.pop("result", None)
            image = None

            if raw:
                try:
                    if len(raw) > MAX_BYTES:
                        raise ValueError(t("Please choose an image smaller than 10 MB."))
                    image = decode_image(raw)
                    st.caption(
                        t(
                            "{name} · {width} × {height} pixels",
                            name=uploaded.name,
                            width=image.width,
                            height=image.height,
                        )
                    )
                except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
                    st.error(
                        t("Unable to read this image. Use a valid PNG/JPG smaller than 10 MB.")
                    )
            if st.button(t("Analyze image"), type="primary", disabled=image is None):
                st.session_state.pop("result", None)
                try:
                    with st.spinner(t("Analyzing image…")):
                        st.session_state["result"] = prediction_request(
                            raw,
                            uploaded.name,
                            "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg",
                        )
                except requests.Timeout:
                    st.error(t("Analysis took too long. Please try again."))
                except requests.RequestException:
                    st.error(
                        t("The analysis service is unavailable. Please try again in a moment.")
                    )
                except ValueError:
                    st.error(t("The service returned an incomplete result. Please try again."))
        result = preview_response() if preview else st.session_state.get("result")
        display_image = image
        if preview:
            try:
                with st.spinner(t("Loading example image…")):
                    display_image, overlay = preview_images()
                result["gradcam_heatmap"] = overlay
            except (requests.RequestException, OSError, ValueError):
                display_image = None
                st.warning(
                    t(
                        "The example image could not load. Set PREVIEW_XRAY_PATH to "
                        "a local X-ray file for offline preview."
                    )
                )
        render_images(display_image, result, image_focus, preview)
        if preview and display_image is not None:
            with compact_section("sample-credit"):
                st.caption(
                    t(
                        "[Sample X-ray: Mikael Häggström · CC0]({source}). Overlay and scores are "
                        "illustrative, not model results for this image.",
                        source=SAMPLE_SOURCE,
                    )
                )

    with right:
        with compact_section("review"):
            render_results(result, preview, image_focus)

    with compact_section("footer"):
        st.divider()
        st.caption(
            t(
                "Educational prototype. AI outputs are for reference and "
                "must not be used for clinical diagnosis."
            )
        )


if __name__ == "__main__":
    main()
