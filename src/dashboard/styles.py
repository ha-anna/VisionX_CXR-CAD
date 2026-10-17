"""VisionX display themes. All visual colors are defined in PALETTES."""

import hashlib
import html
from contextlib import contextmanager

import streamlit as st

PALETTES = {
    "light": {
        "page": "#F2F6F8",
        "surface": "#FFFFFF",
        "soft": "#EAF1F4",
        "text": "#172E40",
        "muted": "#536B7B",
        "line": "#D2DFE5",
        "accent": "#087F83",
        "hover": "#06676B",
        "on-accent": "#FFFFFF",
        "track": "#DDE8EC",
        "bar": "#087F83",
        "chip": "#E4F3F2",
        "chip-text": "#075E62",
        "disabled": "#DDE6EA",
        "disabled-text": "#556B78",
        "warning-bg": "#FFF5DF",
        "warning-text": "#75500E",
        "error-bg": "#FDEEF0",
        "error-text": "#932C40",
        "success-bg": "#EAF5EF",
        "success-text": "#226044",
        "info-bg": "#EAF0F4",
        "info-text": "#405B6B",
    },
    "dark": {
        "page": "#0C1822",
        "surface": "#132632",
        "soft": "#1A3341",
        "text": "#E5F0F4",
        "muted": "#A8BFCC",
        "line": "#35505F",
        "accent": "#60D2CB",
        "hover": "#8DE3DD",
        "on-accent": "#082F36",
        "track": "#294451",
        "bar": "#60D2CB",
        "chip": "#1A4145",
        "chip-text": "#9AE7DD",
        "disabled": "#293E4A",
        "disabled-text": "#A1B6C1",
        "warning-bg": "#352C1B",
        "warning-text": "#F1D193",
        "error-bg": "#37232D",
        "error-text": "#FFB6C3",
        "success-bg": "#19372E",
        "success-text": "#A4DFC0",
        "info-bg": "#1B2D39",
        "info-text": "#B9CEDB",
    },
}

CSS = """
header[data-testid="stHeader"] {display:none;}
[data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background:var(--vx-page);color:var(--vx-text);
}
[data-testid="stMainBlockContainer"], .block-container {
    max-width:960px !important;width:100%;margin-inline:auto;
    box-sizing:border-box;padding:1.5rem 1.25rem 2rem;
}
[data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stWidgetLabel"], [data-testid="stCheckbox"] label,
[data-testid="stRadio"] label, [data-testid="stSpinner"],
[data-testid="stUploadedFile"] {color:var(--vx-text);}
[data-testid="stMain"] h1 {font-weight:600;letter-spacing:-.04em;}
/* Use Streamlit's native heading sizes and hierarchy. */
.vx-title a {color:var(--vx-text);text-decoration:none;}
.vx-title a:hover {color:var(--vx-accent);}
[data-testid="stMain"] .vx-caption {
    color:var(--vx-muted) !important;font-size:.82rem;line-height:1.5;margin:0;
    display:block;box-sizing:border-box;max-width:100%;
    white-space:normal;overflow-wrap:anywhere;
}
.st-key-language-toolbar button {font-size:.8rem;}
/* One outline per main section; no native border underneath it. */
.st-key-vx-compact-upload,
.st-key-vx-compact-image-panel,
.st-key-vx-compact-review {
    border:1px solid var(--vx-line);
    border-radius:10px;
    padding:1rem;
    background:var(--vx-surface);
    box-sizing:border-box;
}
.st-key-vx-compact-explanation {padding-top:.5rem;}
.st-key-vx-compact-sample-credit p {font-size:.75rem;color:var(--vx-muted) !important;line-height:1.5;}
/* Native layout, spacing and borders are left intact. Only colors change. */
[data-testid="stExpander"] details {background:var(--vx-surface);border-color:var(--vx-line);}
/* Override only colors; native expander dimensions and padding stay intact. */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details[open] > summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:active {
    background:var(--vx-surface) !important;
    color:var(--vx-text) !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] summary svg {
    color:var(--vx-text) !important;
}
/* Keep expanded content inside its own padding and in normal flow. */
[data-testid="stExpanderDetails"] {
    box-sizing:border-box;
    padding:1rem !important;
    min-width:0;
    height:auto;
    max-height:none;
}
[data-testid="stExpanderDetails"] [data-testid="stMarkdown"],
[data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] {
    min-width:0;
    max-width:100%;
    box-sizing:border-box;
    white-space:normal;
    overflow-wrap:anywhere;
}
[data-testid="stExpanderDetails"] .vx-caption {
    margin:0 !important;
    height:auto;
    max-height:none;
    line-height:1.5;
}
[data-testid="stFileUploaderDropzone"] {background:var(--vx-soft);border-color:var(--vx-line);}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p {color:var(--vx-text) !important;opacity:1;}
[data-testid="stFileUploaderDropzone"] small {color:var(--vx-muted) !important;}
[data-testid="stBaseButton-primary"] {background:var(--vx-accent);color:var(--vx-on-accent);border-color:var(--vx-accent);}
[data-testid="stBaseButton-primary"] p {color:var(--vx-on-accent) !important;}
[data-testid="stBaseButton-primary"]:hover:not(:disabled) {background:var(--vx-hover);border-color:var(--vx-hover);}
[data-testid="stBaseButton-secondary"] {background:var(--vx-surface);color:var(--vx-text);border-color:var(--vx-line);}
[data-testid="stBaseButton-secondary"] p {color:var(--vx-text) !important;}
[data-testid="stBaseButton-secondary"]:hover:not(:disabled) {background:var(--vx-soft);border-color:var(--vx-accent);}
[data-testid="stMain"] button:disabled {background:var(--vx-disabled);border-color:var(--vx-line);color:var(--vx-disabled-text);}
[data-testid="stMain"] button:disabled p {color:var(--vx-disabled-text) !important;}
button:focus-visible, summary:focus-visible {outline:2px solid var(--vx-accent);outline-offset:2px;}
[data-testid="stImageCaption"] {color:var(--vx-muted);}
hr {border-color:var(--vx-line);}
h3 {padding-bottom: 20.4rem;}
.vx-score, .vx-axis-row {display:grid;grid-template-columns:145px minmax(40px,1fr) 42px;gap:10px;align-items:center;}
.vx-score {font-size:13px;margin:10px 0;}
.vx-track {height:7px;background:var(--vx-track);border-radius:3px;overflow:hidden;}
.vx-fill {height:100%;background:var(--vx-bar);border-radius:3px;}
.vx-value {text-align:right;font-variant-numeric:tabular-nums;}
.vx-axis {display:flex;justify-content:space-between;font-size:11px;color:var(--vx-muted);margin-bottom:4px;}
.vx-chips {display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 12px;}
.vx-chip {background:var(--vx-chip);color:var(--vx-chip-text);border:1px solid var(--vx-line);padding:4px 8px;border-radius:5px;font-size:13px;}
@media(max-width:640px) {
    [data-testid="stMainBlockContainer"], .block-container {padding:1rem .75rem 2rem;}
    .vx-score, .vx-axis-row {grid-template-columns:120px minmax(30px,1fr) 38px;gap:7px;}
}
"""


def apply_styles(image_focus=False):
    theme = st.session_state.get("ui_theme", "light")
    palette = PALETTES.get(theme, PALETTES["light"])
    variables = ";".join(f"--vx-{key}:{value}" for key, value in palette.items())
    css = ":root{" + variables + ";}" + CSS
    # Image focus stacks the content; it does not widen the page.
    for kind in ("warning", "error", "success", "info"):
        selector = f'[class*="st-key-vx-notice-{kind}-"] [data-testid="stAlert"]'
        css += f"""
        {selector}, {selector} > div {{background:var(--vx-{kind}-bg);border-radius:10px;}}
        {selector} {{border:1px solid var(--vx-line);border-left:3px solid var(--vx-{kind}-text);}}
        {selector} [data-testid="stMarkdownContainer"], {selector} p,
        {selector} svg, {selector} a {{color:var(--vx-{kind}-text) !important;}}
        """
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)


@contextmanager
def compact_section(name):
    with st.container(key=f"vx-compact-{name}", border=False):
        yield


def notice(kind, body):
    if kind not in {"warning", "error", "success", "info"}:
        raise ValueError(f"Unsupported notice type: {kind}")
    identity = hashlib.sha256(str(body).encode()).hexdigest()[:12]
    with st.container(key=f"vx-notice-{kind}-{identity}"):
        getattr(st, kind)(body)


def caption(body):
    """Plain explanatory text with explicit theme contrast in both modes."""
    st.markdown(
        '<div class="vx-caption">' + html.escape(str(body)) + "</div>",
        unsafe_allow_html=True,
    )
