from contextlib import contextmanager

import streamlit as st


def apply_styles(image_focus=False):
    st.markdown(
        """
    <style>
    .block-container {max-width:1600px;padding-top:2rem;padding-bottom:2rem;}
    [class*="st-key-vx-compact-"] {max-width:1100px;width:100%;margin-inline:auto;}
    [class*="st-key-vx-compact-upload"] {padding-bottom:1rem;}
    [class*="st-key-vx-compact-review"] {padding-top:1.25rem;}
    [class*="st-key-vx-compact-explanation"] {padding-top:.5rem;}
    h1 {font-weight:600;letter-spacing:-0.04em;margin-bottom:0;}
    .vx-title a {color:inherit;text-decoration:none;}
    .vx-title a:hover {text-decoration:underline;text-underline-offset:6px;}
    [data-testid="stFileUploader"] {border-radius:10px;}
    .vx-score {display:grid;grid-template-columns:145px minmax(20px,1fr) 42px;
        gap:10px;align-items:center;margin:10px 0;font-size:13px;}
    .vx-track {height:9px;background:rgba(128,128,128,.13);border-radius:3px;}
    .vx-fill {height:100%;border-radius:3px;}
    .vx-red {background:#d44b53;}.vx-yellow {background:#c99a25;}
    .vx-green {background:#43876a;}
    .vx-value {text-align:right;font-variant-numeric:tabular-nums;}
    .vx-axis {display:flex;justify-content:space-between;margin:0 52px 12px 155px;
        font-size:12px;opacity:.7;}
    .vx-legend {display:flex;gap:16px;flex-wrap:wrap;font-size:12px;margin-top:18px;}
    .vx-legend span {display:inline-flex;align-items:center;gap:6px;}
    .vx-swatch {display:inline-block;width:8px;height:8px;border-radius:2px;}
    .vx-chips {display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px;}
    .vx-chip {border:1px solid rgba(128,128,128,.3);padding:4px 9px;
        border-radius:5px;font-size:13px;}
    @media(max-width:640px){.vx-score{grid-template-columns:125px minmax(20px,1fr) 38px;gap:7px;}
        .vx-axis{margin-left:132px;margin-right:45px;}}
    </style>
    """,
        unsafe_allow_html=True,
    )

    if not image_focus:
        st.markdown("<style>.block-container {max-width:1100px;}</style>", unsafe_allow_html=True)


@contextmanager
def compact_section(name):
    """Constrain controls/results without shrinking the image viewer."""
    with st.container(key=f"vx-compact-{name}"):
        yield
