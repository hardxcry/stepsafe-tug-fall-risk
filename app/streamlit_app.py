"""StepSafe: Timed Up and Go analysis from an ordinary video.

Run:  streamlit run app/streamlit_app.py
Research prototype. Not a medical device and not for clinical use.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tugrisk.simulate import Protocol, simulate
from tugrisk.tug import PHASES, STEADI_CUTOFF_S, TugError, analyse

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"sit_to_stand": "#0ea5e9", "walk_out": "#10b981", "turn": "#f59e0b",
          "walk_back": "#14b8a6", "turn_to_sit": "#8b5cf6"}
NAMES = {"sit_to_stand": "Sit to stand", "walk_out": "Walk out", "turn": "Turn",
         "walk_back": "Walk back", "turn_to_sit": "Turn and sit"}

st.set_page_config(page_title="StepSafe", page_icon="🚶", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#042f2e 0%,#0f766e 55%,#14b8a6 100%);
       padding: 1.6rem 2rem; border-radius: 18px; color:#fff; margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size:2rem; color:#fff;} .hero p {margin:.3rem 0 0; opacity:.9;}
.pill {display:inline-block; padding:.3rem .9rem; border-radius:999px; font-weight:600;}
.hi {background:#fee2e2; color:#991b1b;} .lo {background:#dcfce7; color:#166534;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🚶 StepSafe</h1><p>Times the CDC Timed Up and Go test from '
            'video, splits it into phases and measures gait. No wearables, no markers.</p></div>',
            unsafe_allow_html=True)


def show_result(r, frames=None):
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=r.total_time, number={"suffix": " s", "font": {"size": 52}},
            gauge={"axis": {"range": [0, 30]},
                   "bar": {"color": "#b91c1c" if r.increased_risk else "#15803d"},
                   "steps": [{"range": [0, STEADI_CUTOFF_S], "color": "rgba(34,197,94,.18)"},
                             {"range": [STEADI_CUTOFF_S, 30], "color": "rgba(239,68,68,.18)"}],
                   "threshold": {"line": {"color": "#111", "width": 3}, "value": STEADI_CUTOFF_S}}))
        g.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g, use_container_width=True)
        tag = ('<span class="pill hi">12 s or more: increased fall risk (CDC STEADI)</span>'
               if r.increased_risk else '<span class="pill lo">Under 12 s: within normal range</span>')
        st.markdown(tag, unsafe_allow_html=True)
        c = st.columns(3)
        c[0].metric("Steps", r.steps)
        c[1].metric("Cadence", f"{r.cadence_spm:.0f}/min")
        c[2].metric("Trunk lean", f"{r.trunk_lean_mean_deg:.0f}°", f"±{r.trunk_lean_sd_deg:.0f}° sway",
                    delta_color="off")
        st.caption(f"Person tracked in {r.detection_rate:.0%} of frames. The simulation study found "
                   "this method reads about 0.5 s short on average; treat times near 12 s with care.")
    with right:
        fig = go.Figure()
        for k in PHASES:
            a, b = r.phases[k]
            fig.add_trace(go.Bar(x=[b - a], y=["Test"], base=a - r.start, orientation="h",
                                 name=f"{NAMES[k]} ({b - a:.1f} s)", marker_color=COLORS[k]))
        fig.update_layout(barmode="overlay", height=150, margin=dict(l=10, r=10, t=10, b=30),
                          xaxis_title="Seconds from start", yaxis_visible=False,
                          legend=dict(orientation="h", y=1.6), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        sig = r.signals
        fig = go.Figure(go.Scatter(x=sig["t"], y=sig["posture"], line=dict(color="#0f766e", width=2),
                                   name="Posture angle"))
        for k in PHASES:
            fig.add_vrect(x0=r.phases[k][0], x1=r.phases[k][1], fillcolor=COLORS[k], opacity=.15,
                          line_width=0)
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=30), xaxis_title="Time (s)",
                          yaxis_title="Hip-knee angle (deg)", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("About 100° seated, 170° standing. Shaded bands are the detected phases.")
    if frames:
        st.image(frames, caption="Tracked skeleton across the test", use_container_width=True)


tab_video, tab_sim, tab_val, tab_about = st.tabs(["Analyse a video", "Try a simulated patient",
                                                  "Validation", "About"])

with tab_video:
    st.markdown("**How to record:** one person only in frame, whole body visible, camera still and "
                "side-on about 4 m away. Start seated, stand, walk 3 m, turn, walk back, sit down.")
    up = st.file_uploader("Video file", type=["mp4", "mov", "webm", "avi", "mkv"])
    a, b = st.columns(2)
    start_s = a.number_input("Clip starts at (s)", 0.0, 3600.0, 0.0, 0.5)
    end_s = b.number_input("Clip ends at (s, 0 = end of video)", 0.0, 3600.0, 0.0, 0.5)
    if up and st.button("Analyse", type="primary"):
        import cv2
        from tugrisk.pose import detect, draw_skeleton, select_subject
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
            tmp.write(up.getbuffer())
        bar = st.progress(0.0, "Estimating pose...")
        try:
            cands, fps = detect(tmp.name, progress=lambda f: bar.progress(f, "Estimating pose..."))
            bar.empty()
            i0 = int(start_s * fps)
            i1 = int(end_s * fps) if end_s > 0 else len(cands)
            seq = select_subject(cands[i0:i1], fps)
            r = analyse(seq)
            cap, tiles = cv2.VideoCapture(tmp.name), []
            for t in np.linspace(r.start, r.end, 8):
                cap.set(cv2.CAP_PROP_POS_MSEC, (start_s + t) * 1000)
                ok, f = cap.read()
                if ok:
                    draw_skeleton(f, seq.image[min(len(seq.image) - 1, int(round(t * fps)))])
                    tiles.append(cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2RGB))
            sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])]) if len(tiles) == 8 else None
            show_result(r, sheet)
        except (TugError, ValueError) as e:
            bar.empty()
            st.error(str(e))

with tab_sim:
    st.caption("A stick-figure patient with known timings, passed through the same analysis.")
    c = st.columns(4)
    speed = c[0].slider("Mobility (1 = frail, 10 = brisk)", 1, 10, 6)
    noise = c[1].slider("Camera noise (cm)", 0, 6, 2)
    fps = c[2].select_slider("Frame rate", [15, 30], 30)
    drop = c[3].slider("Lost frames (%)", 0, 20, 5)
    k = (10 - speed) / 9
    proto = Protocol(sit_to_stand=1.0 + 2.2 * k, walk_out=2.0 + 5.5 * k, turn=1.0 + 2.5 * k,
                     walk_back=2.0 + 5.5 * k, turn_to_sit=2.0 + 3.5 * k, cadence_spm=115 - 40 * k)
    try:
        r = analyse(simulate(proto, fps=fps, noise_m=noise / 100, dropout=drop / 100, seed=7))
        show_result(r)
        st.info(f"True time {proto.total:.1f} s. Measured {r.total_time:.1f} s "
                f"(error {r.total_time - proto.total:+.1f} s).")
    except TugError as e:
        st.error(str(e))

with tab_val:
    mp_ = ROOT / "reports" / "metrics.json"
    if mp_.exists():
        m = json.loads(mp_.read_text())
        ref = next(c for c in m["conditions"] if c["fps"] == 30 and c["noise_cm"] == 2)
        c = st.columns(4)
        c[0].metric("Mean error", f"{ref['bias_s']:+.2f} s")
        c[1].metric("Mean absolute error", f"{ref['mae_s']:.2f} s")
        c[2].metric("Sensitivity at 12 s", f"{ref['sensitivity']:.1%}")
        c[3].metric("Specificity at 12 s", f"{ref['specificity']:.1%}")
        st.caption("Simulation study, 30 fps, 2 cm landmark noise, 200 simulated tests. "
                   "Simulated data only: this is not clinical validation.")
        st.image(str(ROOT / "reports/figures/simulation.png"))
        st.dataframe(pd.DataFrame([{"fps": x["fps"], "noise (cm)": x["noise_cm"], "bias (s)": x["bias_s"],
                                    "MAE (s)": x["mae_s"], "sensitivity": x["sensitivity"],
                                    "specificity": x["specificity"]} for x in m["conditions"]]).round(3),
                     use_container_width=True, hide_index=True)
    if (ROOT / "reports/figures/real_case.jpg").exists():
        st.subheader("Real-video case study")
        st.image(str(ROOT / "reports/figures/real_case.jpg"),
                 caption="CDC STEADI demonstration video (public domain)")

with tab_about:
    st.markdown("""
**The test.** In the Timed Up and Go, a person rises from a chair, walks 3 metres, turns, walks
back and sits down. The CDC STEADI programme treats 12 seconds or more as a sign of increased
fall risk in older adults.

**The method.** MediaPipe Pose finds 33 body landmarks per frame. Joint angles show when the
person stands and sits; travel across the image locates the turn; ankle separation counts steps.

**Limits.** Validated on simulated movement and one demonstration video only. It expects a single
person in view; a helper walking alongside confuses the tracking. It is not a medical device.
""")
