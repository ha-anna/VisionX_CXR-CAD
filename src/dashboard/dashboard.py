import hashlib

import requests
import streamlit as st
from api_client import prediction_request
from components import render_images, render_results
from image_utils import decode_image
from PIL import Image, UnidentifiedImageError
from preview import SAMPLE_SOURCE, preview_images, preview_response
from settings import MAX_BYTES
from styles import apply_styles, compact_section


def main():
    st.set_page_config(page_title="VisionX", page_icon="🩻", layout="wide")
    apply_styles(True)

    with compact_section("header"):
        st.markdown(
            '<h1 class="vx-title"><a href="https://github.com/ha-anna/VisionX_CXR-CAD" '
            'target="_blank" rel="noopener noreferrer" '
            'aria-label="VisionX GitHub repository">VisionX ↗</a></h1>',
            unsafe_allow_html=True,
        )
        st.caption("Upload a chest X-ray, analyze it, and review the model's findings.")
        image_focus = st.checkbox(
            "Enlarge image viewer",
            value=False,
            help="Give images more space and place results below them.",
        )
        with st.expander("Demo options"):
            preview = st.toggle("Show illustrative results", value=False)
            st.caption("Preview the interface without sending an image for analysis.")
        if preview:
            st.warning(
                "Preview mode · Scores are illustrative and are not predictions for your image."
            )
        st.divider()

    apply_styles(image_focus)
    left, right = (
        (st.container(), st.container()) if image_focus else st.columns([1, 1.05], gap="large")
    )

    with left:
        with compact_section("upload"):
            st.subheader("Upload image")
            st.caption("PNG or JPEG · Up to 10 MB")
            uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
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
                        raise ValueError("Please choose an image smaller than 10 MB.")
                    image = decode_image(raw)
                    st.caption(f"{uploaded.name} · {image.width} × {image.height} pixels")
                except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
                    st.error("Unable to read this image. Use a valid PNG/JPEG smaller than 10 MB.")
            if st.button("Analyze image", type="primary", disabled=image is None or preview):
                st.session_state.pop("result", None)
                try:
                    with st.spinner("Analyzing image…"):
                        st.session_state["result"] = prediction_request(
                            raw,
                            uploaded.name,
                            "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg",
                        )
                except requests.Timeout:
                    st.error("Analysis took too long. Please try again.")
                except requests.RequestException:
                    st.error("The analysis service is unavailable. Please try again in a moment.")
                except ValueError:
                    st.error("The service returned an incomplete result. Please try again.")
        result = preview_response() if preview else st.session_state.get("result")
        display_image = image
        if preview:
            try:
                with st.spinner("Loading example image…"):
                    display_image, overlay = preview_images()
                result["gradcam_heatmap"] = overlay
            except (requests.RequestException, OSError, ValueError):
                display_image = None
                st.warning(
                    "The example image could not load. Set PREVIEW_XRAY_PATH to a local "
                    "X-ray file for offline preview."
                )
        render_images(display_image, result, image_focus, preview)
        if preview and display_image is not None:
            with compact_section("sample-credit"):
                st.caption(
                    f"[Sample X-ray: Mikael Häggström · CC0]({SAMPLE_SOURCE}). "
                    "Overlay and scores are illustrative, not model results for this image."
                )

    with right:
        with compact_section("review"):
            render_results(result, preview, image_focus)

    with compact_section("footer"):
        st.divider()
        st.caption(
            "Educational prototype. AI outputs are for reference "
            "and must not be used for clinical diagnosis."
        )


if __name__ == "__main__":
    main()
