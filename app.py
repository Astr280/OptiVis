"""
🩺 OptiVis — AI-Powered Diabetic Retinopathy Detection
=====================================================
OptiVis — AI-Powered Retinal Diagnostics
Streamlit application with:
  • Expert ResNet-50 Hugging Face inference
  • Grad-CAM heatmap visualisation
  • Risk interpretation & recommendations
  • Upload history panel
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import io, time, os

from preprocessing import load_image, get_model_input, preprocess_image
from model import predict, CLASS_NAMES
from gradcam import make_gradcam_figure
from history import add_record, get_history, clear_history


# ─────────────────────────────────────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OptiVis — AI Retinal Diagnostics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  (dark glassmorphism theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ── Root variables ── */
:root {
    --bg-deep:    #020817;
    --bg-card:    rgba(15, 25, 50, 0.85);
    --bg-glass:   rgba(255,255,255,0.04);
    --accent:     #00D4FF;  /* Matches OptiVis Logo Teal */
    --accent2:    #1B3E63;  /* Matches OptiVis Logo Dark Blue */
    --success:    #00E676;
    --warning:    #FFD600;
    --danger:     #FF3D57;
    --text-main:  #E8EAF6;
    --text-muted: #8892B0;
    --border:     rgba(0, 212, 255, 0.15);
}

/* ── App background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0A1628 0%, #020817 60%),
                radial-gradient(ellipse at 80% 100%, #0D1B3E 0%, transparent 60%);
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060F25 0%, #0A1628 100%) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text-muted); }

/* ── Hide default header/footer ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

/* ── Glassmorphism card ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1rem;
}

/* ── Metric badge ── */
.metric-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-green  { background: rgba(0,230,118,0.15); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }
.badge-yellow { background: rgba(255,214,0,0.15);  color: #FFD600; border: 1px solid rgba(255,214,0,0.3); }
.badge-red    { background: rgba(255,61,87,0.15);  color: #FF3D57; border: 1px solid rgba(255,61,87,0.3); }
.badge-purple { background: rgba(123,97,255,0.15); color: #7B61FF; border: 1px solid rgba(123,97,255,0.3);}

/* ── Progress bar override ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 100px;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0, 212, 255, 0.3) !important;
    border-radius: 12px !important;
    background: rgba(0, 212, 255, 0.03) !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 212, 255, 0.7) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #7B61FF);
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.6rem !important;
    letter-spacing: 0.03em;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(0, 212, 255, 0.5);
}

/* ── Info / warning boxes ── */
.stAlert {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Toggle / checkbox ── */
.stCheckbox label { color: var(--text-muted) !important; font-size: 0.88rem; }

/* ── History item ── */
.history-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 6px;
    font-size: 0.82rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Severity metadata
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_META = {
    "No DR": {
        "icon": "✅",
        "badge": "badge-green",
        "risk": "Low Risk",
        "color": "#00E676",
        "description": "No signs of diabetic retinopathy detected. The retina appears healthy.",
        "findings": "No microaneurysms, haemorrhages, or exudates visible.",
        "recommendation": "Continue regular annual eye examinations. Maintain good blood sugar control.",
        "alert_level": "🟢 SAFE",
        "action": "Routine annual check-up recommended.",
    },
    "Mild": {
        "icon": "⚠️",
        "badge": "badge-yellow",
        "risk": "Mild Risk",
        "color": "#FFD600",
        "description": "Mild non-proliferative diabetic retinopathy (NPDR) detected.",
        "findings": "Small micro-aneurysms present — tiny balloon-like swellings in blood vessels.",
        "recommendation": "Consult an ophthalmologist within 12 months. Optimise blood sugar and blood pressure.",
        "alert_level": "🟡 MONITOR",
        "action": "Schedule ophthalmology review within 6–12 months.",
    },
    "Moderate": {
        "icon": "🔶",
        "badge": "badge-yellow",
        "risk": "Moderate Risk",
        "color": "#FF8C00",
        "description": "Moderate non-proliferative diabetic retinopathy (NPDR) detected.",
        "findings": "Multiple micro-aneurysms, dot & blot haemorrhages, and possible hard exudates.",
        "recommendation": "Ophthalmologist consultation within 6 months. Consider laser therapy evaluation.",
        "alert_level": "🟠 MODERATE RISK",
        "action": "Seek specialist review within 3–6 months.",
    },
    "Severe": {
        "icon": "🚨",
        "badge": "badge-red",
        "risk": "High Risk",
        "color": "#FF3D57",
        "description": "Severe non-proliferative diabetic retinopathy (NPDR) detected.",
        "findings": "Extensive retinal haemorrhages, venous beading, and intraretinal microvascular abnormalities (IRMA).",
        "recommendation": "Urgent ophthalmology referral. High risk of progression to vision-threatening PDR.",
        "alert_level": "🔴 HIGH RISK",
        "action": "Urgent ophthalmology referral — within 1 month.",
    },
    "Proliferative DR": {
        "icon": "🆘",
        "badge": "badge-red",
        "risk": "Critical Risk",
        "color": "#FF0040",
        "description": "Proliferative diabetic retinopathy (PDR) detected — most advanced stage.",
        "findings": "Neovascularisation (new fragile blood vessels) on the retina/optic disc. High risk of vitreous haemorrhage and retinal detachment.",
        "recommendation": "Immediate referral to retinal specialist. Laser photocoagulation or anti-VEGF injections may be required.",
        "alert_level": "🆘 CRITICAL",
        "action": "IMMEDIATE specialist referral. Do not delay.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper rendering functions
# ─────────────────────────────────────────────────────────────────────────────

def render_confidence_chart(probs: np.ndarray, class_idx: int) -> go.Figure:
    """Horizontal bar chart of class probabilities."""
    names  = CLASS_NAMES[::-1]
    values = (probs * 100).tolist()[::-1]
    colors = ["#FF0040", "#FF3D57", "#FF8C00", "#FFD600", "#00E676"][::-1]

    highlight = [
        f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.9)"
        if (len(CLASS_NAMES) - 1 - i) == class_idx
        else f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.35)"
        for i, c in enumerate(colors)
    ]

    fig = go.Figure(
        go.Bar(
            x          = values,
            y          = names,
            orientation = "h",
            marker_color = highlight,
            marker_line_width = 0,
            text       = [f"{v:.1f}%" for v in values],
            textposition = "outside",
            textfont   = dict(color="rgba(232,234,246,0.85)", size=12),
        )
    )
    fig.update_layout(
        xaxis = dict(range=[0, 115], showgrid=False, zeroline=False,
                     ticksuffix="%", color="rgba(136,146,176,0.7)", tickfont_size=11),
        yaxis = dict(showgrid=False, color="#E8EAF6", tickfont_size=13),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = dict(l=0, r=60, t=10, b=10),
        height        = 220,
        showlegend    = False,
    )
    return fig


def render_gauge(confidence: float, color: str) -> go.Figure:
    """Semicircular confidence gauge."""
    fig = go.Figure(
        go.Indicator(
            mode  = "gauge+number",
            value = round(confidence * 100, 1),
            number = dict(suffix="%", font=dict(size=28, color=color, family="Space Grotesk")),
            gauge = dict(
                axis      = dict(range=[0, 100], tickcolor="rgba(136,146,176,0.5)", tickfont_size=10),
                bar       = dict(color=color, thickness=0.22),
                bgcolor   = "rgba(255,255,255,0.04)",
                borderwidth = 0,
                steps     = [
                    dict(range=[0, 40],   color="rgba(255,255,255,0.04)"),
                    dict(range=[40, 70],  color="rgba(255,255,255,0.03)"),
                    dict(range=[70, 100], color="rgba(255,255,255,0.02)"),
                ],
                threshold = dict(line=dict(color=color, width=3), thickness=0.75, value=confidence * 100),
            ),
            title = dict(text="Confidence", font=dict(color="rgba(136,146,176,0.8)", size=13)),
        )
    )
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        font_color    = "#E8EAF6",
        height        = 190,
        margin        = dict(l=20, r=20, t=30, b=10),
    )
    return fig


def speak_result(text: str):
    """Text-to-speech output (non-blocking)."""
    try:
        import pyttsx3, threading
        def _speak():
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=_speak, daemon=True).start()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Logo Integration ──
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem 0 0.5rem;">
                <div style="font-size:2.5rem;">👁️</div>
                <h2 style="color:#00D4FF;margin:0;font-family:'Space Grotesk',sans-serif;">OptiVis</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown(
        """
        <div style="text-align:center;">
            <p style="color:#8892B0;font-size:0.75rem;margin:0;letter-spacing:0.05em;">AI-POWERED RETINAL DIAGNOSTICS</p>
        </div>
        <hr style="border-color:rgba(0,212,255,0.15);margin:1.2rem 0;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### ⚙️ Settings")
    enable_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
    enable_voice   = st.checkbox("Voice Output", value=False)
    show_history   = st.checkbox("Show Upload History", value=True)

    st.markdown("<hr style='border-color:rgba(0,212,255,0.1)'>", unsafe_allow_html=True)
    st.markdown("##### 📚 About")
    st.markdown(
        """
<p style="font-size:0.8rem;color:#8892B0;line-height:1.6;">
<b>OptiVis</b> uses a custom <b style="color:#00D4FF">EfficientNet-B0</b> model, 
fine-tuned on the <b style="color:#00D4FF">APTOS 2019</b> dataset for expert precision.<br><br>
Powered by your high-accuracy custom training weights.
</p>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:rgba(0,212,255,0.1)'>", unsafe_allow_html=True)
    st.markdown("##### 🏷️ DR Severity Scale")
    for cls, meta in SEVERITY_META.items():
        st.markdown(
            f"<span style='color:{meta['color']};font-size:0.82rem;'>"
            f"{meta['icon']} <b>{cls}</b></span> "
            f"<span style='color:#8892B0;font-size:0.76rem;'>— {meta['risk']}</span>",
            unsafe_allow_html=True,
        )

    # History panel
    if show_history:
        st.markdown("<hr style='border-color:rgba(0,212,255,0.1)'>", unsafe_allow_html=True)
        history = get_history()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("##### 🕘 Recent Analyses")
        with col2:
            if history and st.button("🗑", key="clear_hist"):
                clear_history()
                st.rerun()

        history = get_history()
        if not history:
            st.markdown(
                "<p style='color:#8892B0;font-size:0.78rem;'>No analyses yet.</p>",
                unsafe_allow_html=True,
            )
        else:
            for rec in history:
                color = SEVERITY_META.get(rec["class"], {}).get("color", "#E8EAF6")
                icon  = SEVERITY_META.get(rec["class"], {}).get("icon", "📷")
                st.markdown(
                    f"""
<div class="history-item">
  <span style="font-size:1.1rem;">{icon}</span>
  <div>
    <div style="color:#E8EAF6;font-weight:600;">{rec['class']}</div>
    <div style="color:#8892B0;">{rec['filename'][:18]}… · {rec['confidence']}%</div>
    <div style="color:#8892B0;font-size:0.72rem;">{rec['timestamp']}</div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Main page header
# ─────────────────────────────────────────────────────────────────────────────

# ── Header Logic ──
logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    # Centered logo for the main page
    col_l, col_r = st.columns([1,1])
    with col_l:
        st.image(logo_path, width=400)
    with col_r:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style=\"font-family:'Space Grotesk',sans-serif;font-weight:800;color:#E8EAF6;margin:0;\">OptiVis</h1>"
            "<p style=\"color:#8892B0;font-size:1.15rem;\">AI-Powered Retinal Diagnostics</p>",
            unsafe_allow_html=True
        )
else:
    st.markdown(
        """
    <div style="text-align:center;padding:2rem 0 1.5rem;">
        <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.8rem;
                   background:linear-gradient(135deg,#00D4FF,#1B3E63);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin:0;">
            OptiVis
        </h1>
        <p style="color:#8892B0;font-size:1.05rem;margin-top:0.4rem;">
            AI-Powered Retinal Diagnostics
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-top:0.8rem;">
        <span class="metric-badge badge-green">✓ EfficientNet-B0</span>
        <span class="metric-badge badge-purple">✓ Grad-CAM XAI</span>
        <span class="metric-badge badge-green">✓ 5-Class Classifier</span>
        <span class="metric-badge badge-purple">✓ Real-time &lt;3s</span>
    </div>
<hr>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Upload section
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<h3 style='color:#E8EAF6;margin-bottom:0.5rem;'>📤 Upload Retinal Fundus Image</h3>",
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Drag & drop a JPG/PNG retinal fundus image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    # ── Load & preprocess ──
    img_rgb       = load_image(uploaded_file)
    model_input   = get_model_input(img_rgb)

    # ── Run inference ──
    with st.spinner("🧠 Running AI inference…"):
        t0 = time.time()
        probs, class_idx, confidence, simulated = predict(img_rgb, model_input)
        elapsed = time.time() - t0

    class_name = CLASS_NAMES[class_idx]
    meta       = SEVERITY_META[class_name]

    # ── Save to history ──
    add_record(uploaded_file.name, img_rgb, class_name, confidence, probs, simulated)

    # ── Voice output ──
    if enable_voice:
        speak_result(
            f"Analysis complete. Detected {class_name} with {round(confidence*100,1)} percent confidence. "
            f"{meta['recommendation']}"
        )


    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # ROW 1:  Image  |  Prediction Dashboard
    # ════════════════════════════════════════════════════════════════════════
    col_img, col_pred = st.columns([1, 1.3], gap="large")

    with col_img:
        st.markdown(
            "<div class='glass-card'>"
            "<h4 style='margin:0 0 0.8rem;color:#00D4FF;'>📷 Uploaded Retinal Image</h4>",
            unsafe_allow_html=True,
        )
        st.image(img_rgb, use_container_width=True, caption=uploaded_file.name)
        st.markdown(
            f"<p style='color:#8892B0;font-size:0.8rem;margin-top:0.4rem;'>"
            f"Size: {img_rgb.shape[1]}×{img_rgb.shape[0]}px · "
            f"Inference time: <b style='color:#00D4FF'>{elapsed:.2f}s</b></p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_pred:
        # ── Main prediction card ──
        st.markdown(
            f"""
<div class="glass-card">
  <h4 style="margin:0 0 0.5rem;color:#00D4FF;">🔬 AI Prediction Result</h4>
  
  <div style="display:flex;align-items:center;gap:12px;margin:1rem 0;">
    <span style="font-size:2.5rem;">{meta['icon']}</span>
    <div>
      <div style="font-size:1.7rem;font-weight:700;color:{meta['color']};
                  font-family:'Space Grotesk',sans-serif;line-height:1.1;">
        {class_name}
      </div>
      <span class="metric-badge {meta['badge']}">{meta['risk']}</span>
    </div>
  </div>
  
  <div style="margin:0.5rem 0 1rem;">
    <div style="color:#8892B0;font-size:0.8rem;margin-bottom:4px;">
      Confidence Score
    </div>
    <div style="font-size:2rem;font-weight:700;color:{meta['color']};
                font-family:'Space Grotesk',sans-serif;">
      {round(confidence*100,1)}%
    </div>
  </div>
  
  <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:1rem;
              border-left:4px solid {meta['color']};">
    <p style="margin:0;color:#E8EAF6;font-size:0.88rem;line-height:1.6;">
      <b>Findings:</b> {meta['findings']}
    </p>
  </div>

  <div style="margin-top:0.8rem;background:rgba(255,255,255,0.03);border-radius:10px;
              padding:0.9rem;border-left:4px solid #7B61FF;">
    <p style="margin:0;color:#E8EAF6;font-size:0.88rem;line-height:1.6;">
      <b>Recommendation:</b> {meta['recommendation']}
    </p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ── Gauge ──
        st.plotly_chart(render_gauge(confidence, meta["color"]), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # ROW 2:  Probability chart  |  Alert banner
    # ════════════════════════════════════════════════════════════════════════
    col_chart, col_alert = st.columns([1.4, 1], gap="large")

    with col_chart:
        st.markdown(
            "<div class='glass-card'>"
            "<h4 style='margin:0 0 1rem;color:#00D4FF;'>📊 Class Probability Distribution</h4>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(render_confidence_chart(probs, class_idx), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_alert:
        st.markdown(
            f"""
<div class="glass-card" style="height:100%;border-color:rgba({
    '0,230,118' if class_idx == 0 else
    '255,214,0' if class_idx <= 2 else
    '255,61,87'
},0.35);">
  <h4 style="color:#00D4FF;margin:0 0 1rem;">🚦 Risk Alert System</h4>

  <div style="text-align:center;padding:1.2rem;border-radius:12px;
              background:rgba({'0,230,118' if class_idx==0 else '255,214,0' if class_idx<=2 else '255,61,87'},0.1);">
    <div style="font-size:2.2rem;font-weight:800;color:{meta['color']};
                font-family:'Space Grotesk',sans-serif;">
      {meta['alert_level']}
    </div>
  </div>

  <div style="margin-top:1rem;">
    <p style="color:#E8EAF6;font-size:0.88rem;line-height:1.6;margin:0;">
      {meta['description']}
    </p>
  </div>

  <div style="margin-top:1rem;padding:0.8rem;background:rgba(255,255,255,0.04);
              border-radius:10px;">
    <b style="color:#7B61FF;font-size:0.82rem;">⚡ Recommended Action</b>
    <p style="color:#E8EAF6;font-size:0.85rem;margin:0.3rem 0 0;line-height:1.5;">
      {meta['action']}
    </p>
  </div>

  <div style="margin-top:1rem;">
    <a href="#" onclick="return false;"
       style="display:block;text-align:center;padding:0.7rem;
              background:linear-gradient(135deg,#00D4FF,#7B61FF);
              color:white;border-radius:10px;font-weight:600;
              text-decoration:none;font-size:0.9rem;
              box-shadow:0 4px 15px rgba(0,212,255,0.3);">
      👨‍⚕️ Consult a Doctor
    </a>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # ROW 3:  Grad-CAM heatmaps
    # ════════════════════════════════════════════════════════════════════════
    if enable_gradcam:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='color:#E8EAF6;margin-bottom:0.5rem;'>🔥 Grad-CAM Explainability</h3>"
            "<p style='color:#8892B0;font-size:0.88rem;margin-bottom:1rem;'>"
            "Gradient-weighted Class Activation Mapping highlights the retinal regions "
            "that most influenced the model's prediction. "
            "<b style='color:#00D4FF'>Warmer colours (red/yellow)</b> = higher relevance.</p>",
            unsafe_allow_html=True,
        )

        with st.spinner("🔥 Generating Grad-CAM heatmap…"):
            cam, heatmap_img, overlay_img = make_gradcam_figure(
                img_rgb, model_input, class_idx, simulated
            )

        gc1, gc2, gc3 = st.columns(3, gap="medium")

        with gc1:
            st.markdown(
                "<div class='glass-card'><h5 style='color:#00D4FF;margin:0 0 0.6rem;'>Original Image</h5>",
                unsafe_allow_html=True,
            )
            st.image(img_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with gc2:
            st.markdown(
                "<div class='glass-card'><h5 style='color:#FF8C00;margin:0 0 0.6rem;'>Activation Heatmap</h5>",
                unsafe_allow_html=True,
            )
            st.image(heatmap_img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with gc3:
            st.markdown(
                "<div class='glass-card'><h5 style='color:#7B61FF;margin:0 0 0.6rem;'>Overlay Visualisation</h5>",
                unsafe_allow_html=True,
            )
            st.image(overlay_img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Heatmap interpretation
        st.markdown(
            f"""
<div class="glass-card" style="border-color:rgba(123,97,255,0.25);">
  <h5 style="color:#7B61FF;margin:0 0 0.8rem;">🧠 Heatmap Interpretation</h5>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;">
    <div style="background:rgba(255,0,64,0.1);border-radius:10px;padding:0.8rem;
                border-left:3px solid #FF0040;">
      <b style="color:#FF3D57;">🔴 Red/Orange zones</b>
      <p style="color:#8892B0;font-size:0.82rem;margin:0.3rem 0 0;">
        Highest-attention regions — likely sites of haemorrhages, 
        exudates, or neovascularisation.
      </p>
    </div>
    <div style="background:rgba(255,214,0,0.1);border-radius:10px;padding:0.8rem;
                border-left:3px solid #FFD600;">
      <b style="color:#FFD600;">🟡 Yellow zones</b>
      <p style="color:#8892B0;font-size:0.82rem;margin:0.3rem 0 0;">
        Moderate attention — possible micro-aneurysms or 
        vascular changes under review.
      </p>
    </div>
    <div style="background:rgba(0,212,255,0.1);border-radius:10px;padding:0.8rem;
                border-left:3px solid #00D4FF;">
      <b style="color:#00D4FF;">🔵 Blue/Green zones</b>
      <p style="color:#8892B0;font-size:0.82rem;margin:0.3rem 0 0;">
        Low-attention areas — likely healthy retinal tissue 
        with no significant features.
      </p>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # Footer disclaimer
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
<div style="text-align:center;padding:1rem;background:rgba(255,255,255,0.02);
            border-radius:12px;border:1px solid rgba(255,255,255,0.05);">
  <p style="color:#8892B0;font-size:0.78rem;margin:0;line-height:1.7;">
    ⚠️ <b>Medical Disclaimer:</b> OptiVis is a screening aid only and is not a substitute 
    for professional medical diagnosis. Always consult a qualified ophthalmologist 
    for clinical decisions. · Expert ResNet-50 Fine-tuned on APTOS 2019
  </p>
  <p style="color:rgba(136,146,176,0.5);font-size:0.72rem;margin:0.3rem 0 0;">
    🏆 Hackathon Demo · AI-powered early detection for accessible healthcare
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

else:
    # ── Landing / empty state ──
    st.markdown(
        """
<div style="text-align:center;padding:4rem 2rem;">
  <div style="font-size:5rem;margin-bottom:1rem;">👁️</div>
  <h2 style="color:#E8EAF6;font-family:'Space Grotesk',sans-serif;margin:0;">
    Upload a Retinal Fundus Image to Begin
  </h2>
  <p style="color:#8892B0;margin-top:0.8rem;max-width:540px;margin-left:auto;margin-right:auto;
            font-size:0.95rem;line-height:1.7;">
    OptiVis will analyse the image using a custom-trained EfficientNet-B0 engine, 
    highly fine-tuned on the APTOS 2019 dataset, providing expert-level 
    classification, confidence score, and Grad-CAM heatmap.
  </p>

  <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;margin-top:2rem;">
    <div class="glass-card" style="max-width:180px;text-align:center;">
      <b style="color:#00D4FF;">EfficientNet-B0</b>
      <p style="color:#8892B0;font-size:0.8rem;margin:0.3rem 0 0;">Custom Training Mode</p>
    </div>
    <div class="glass-card" style="max-width:180px;text-align:center;">
      <div style="font-size:2rem;">🔥</div>
      <b style="color:#7B61FF;">Grad-CAM XAI</b>
      <p style="color:#8892B0;font-size:0.8rem;margin:0.3rem 0 0;">Explainable AI heatmaps</p>
    </div>
    <div class="glass-card" style="max-width:180px;text-align:center;">
      <div style="font-size:2rem;">⚡</div>
      <b style="color:#00E676;">Real-time</b>
      <p style="color:#8892B0;font-size:0.8rem;margin:0.3rem 0 0;">Inference in &lt;3 seconds</p>
    </div>
    <div class="glass-card" style="max-width:180px;text-align:center;">
      <div style="font-size:2rem;">🚦</div>
      <b style="color:#FFD600;">Risk Alerts</b>
      <p style="color:#8892B0;font-size:0.8rem;margin:0.3rem 0 0;">Color-coded severity system</p>
    </div>
  </div>

  <div style="margin-top:2.5rem;padding:1rem 2rem;background:rgba(0,212,255,0.06);
              border-radius:12px;border:1px solid rgba(0,212,255,0.15);
              display:inline-block;max-width:500px;">
    <p style="color:#8892B0;font-size:0.82rem;margin:0;">
      💡 <b style="color:#00D4FF;">Supported formats:</b> JPG, JPEG, PNG retinal fundus images<br>
         Recommended: high-quality fundus photographs (≥256×256 px)
    </p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
