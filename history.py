"""
History & session utilities.
Stores previous analyses in Streamlit's session_state as a list of dicts.
"""

import streamlit as st
from datetime import datetime
import numpy as np


MAX_HISTORY = 10


def _init():
    if "history" not in st.session_state:
        st.session_state["history"] = []


def add_record(
    filename: str,
    img_rgb: np.ndarray,
    class_name: str,
    confidence: float,
    probs: np.ndarray,
    simulated: bool,
):
    _init()
    record = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "filename":  filename,
        "thumbnail": img_rgb,          # store original for display
        "class":     class_name,
        "confidence": round(confidence * 100, 1),
        "probs":      probs.tolist(),
        "simulated":  simulated,
    }
    st.session_state["history"].insert(0, record)
    # Keep only the last MAX_HISTORY
    st.session_state["history"] = st.session_state["history"][:MAX_HISTORY]


def get_history() -> list[dict]:
    _init()
    return st.session_state["history"]


def clear_history():
    st.session_state["history"] = []
