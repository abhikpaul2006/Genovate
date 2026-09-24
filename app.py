"""
app.py — interactive live flood-depth demo (Streamlit)

Run with:  streamlit run app.py

Lets you pick a rainfall value / timestep live and instantly see the
model's predicted depth across the real drainage network — no pre-rendered
video, reacts to input in front of judges.

Run from your `flood prediction` project folder, next to
model.py, graph_model1.npz, cache_model1.npz, flood_gnn_model1.pt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import streamlit as st

from model import FloodGNN

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
ROOT = r"D:\Dataset_Rerelease\Models\Model_1\train"   # same root graph.py used, for static positions
WARMUP = 10
MAX_CUSTOM_STEPS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform(d): return np.sqrt(np.clip(d, 0, None))
def inv_transform(d): return d ** 2

# ----------------------------------------------------------------------
# Cached loaders — run once per app session, not on every slider move
# ----------------------------------------------------------------------
@st.cache_resource
def load_everything():
    g = np.load("graph_model1.npz")
    node_static = torch.tensor(g["node_features"], dtype=torch.float32, device=device)
    src = torch.tensor(g["edge_index"][0], dtype=torch.long, device=device)
    dst = torch.tensor(g["edge_index"][1], dtype=torch.long, device=device)
    N1D, N2D = int(g["N1D"]), int(g["N2D"])
    N = N1D + N2D

    # edge_feat isn't stored directly — rebuilt exactly as train.py/evaluate.py do it
    edge_type_oh = np.eye(3, dtype="float32")[g["edge_type"].astype("int64")]
    efr = g["edge_feat_raw"].astype("float32")
    efr = (efr - efr.mean(0)) / (efr.std(0) + 1e-6)
    edge_feat = torch.tensor(np.concatenate([efr, edge_type_oh], axis=1), dtype=torch.float32, device=device)

    model = FloodGNN(node_in=13 + 4, edge_in=edge_feat.shape[1], hidden=64, layers=4).to(device)
    model.load_state_dict(torch.load("flood_gnn_model1.pt", map_location=device))
    model.eval()

    # real positions from the same file + sort order graph.py used — guaranteed aligned to node rows N1D:
    n2d = pd.read_csv(os.path.join(ROOT, "2d_nodes_static.csv")).sort_values("node_idx").reset_index(drop=True)
    xy = n2d[["position_x", "position_y"]].values.astype("float32")

    cache = np.load("cache_model1.npz")
    all_events = list(cache["events"])
    all_rain_max, all_cum_max = 0.0, 0.0
    for i in range(len(all_events)):
        r = cache[f"rain_{i}"]
        all_rain_max = max(all_rain_max, float(r.max()))
        all_cum_max = max(all_cum_max, float(np.cumsum(r).max()))
    all_rain_max = max(all_rain_max, 1e-6)
    all_cum_max = max(all_cum_max, 1e-6)

    return node_static, src, dst, edge_feat, N1D, N2D, N, model, xy, cache, all_rain_max, all_cum_max


node_static, src, dst, edge_feat, N1D, N2D, N, model, xy, cache, all_rain_max, all_cum_max = load_everything()


def predict_step(prev_depth_transformed, rain_t, cum_t, t_frac):
    dyn = np.zeros((1, N, 4), dtype="float32")
    dyn[0, N1D:, 0] = prev_depth_transformed
    dyn[0, :, 1] = rain_t / all_rain_max
    dyn[0, :, 2] = cum_t / all_cum_max
    dyn[0, :, 3] = t_frac
    static = node_static.unsqueeze(0)
    x = torch.cat([static, torch.tensor(dyn, dtype=torch.float32, device=device)], dim=-1)
    with torch.no_grad():
        pred = model(x, edge_feat, src, dst)
    return pred[0].detach().cpu().numpy()  # (N,), sqrt-depth, all nodes


# ----------------------------------------------------------------------
# Full-rollout computation, cached per SCENARIO (not per slider position).
# This is what makes the slider feel instant: drag it and you're just
# indexing into an array that's already sitting in memory.
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Running full rollout...")
def compute_replay_rollout(event_id):
    depth2d = np.nan_to_num(cache[f"depth_{event_id}"], nan=0.0).astype("float32")
    rain = cache[f"rain_{event_id}"].astype("float32")
    T = depth2d.shape[0]
    cum = np.cumsum(rain)

    depth_full = np.zeros((T, N2D), dtype="float32")
    depth_full[:WARMUP] = depth2d[:WARMUP]  # ground truth, no model call needed

    last_pred = None
    for t in range(WARMUP, T):
        prevd = transform(depth2d[t - 1]) if t <= WARMUP else last_pred
        pred = predict_step(prevd, rain[t], cum[t], t / max(T - 1, 1))
        pred_2d = np.clip(pred[N1D:], 0, None)
        last_pred = pred_2d
        depth_full[t] = inv_transform(pred_2d)
    return depth_full


@st.cache_data(show_spinner="Running what-if scenario...")
def compute_custom_rollout(custom_rain):
    rain = np.full(MAX_CUSTOM_STEPS, custom_rain, dtype="float32")
    cum = np.cumsum(rain)
    depth_full = np.zeros((MAX_CUSTOM_STEPS, N2D), dtype="float32")

    last_pred = np.zeros(N2D, dtype="float32")  # start dry
    for t in range(MAX_CUSTOM_STEPS):
        pred = predict_step(last_pred, rain[t], cum[t], t / max(MAX_CUSTOM_STEPS - 1, 1))
        pred_2d = np.clip(pred[N1D:], 0, None)
        last_pred = pred_2d
        depth_full[t] = inv_transform(pred_2d)
    return depth_full


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.set_page_config(layout="wide")
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container { padding-top: 1.5rem; }

h1 {
    font-weight: 700 !important;
    background: linear-gradient(90deg, #2AA9C4, #6FE3D6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

[data-testid="stMetric"] {
    background-color: #151B2B;
    border: 1px solid #24304A;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #6FE3D6; }

section[data-testid="stSidebar"] {
    background-color: #0B0F1A;
    border-right: 1px solid #1E2636;
}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>🌊 Smart Urban Drainage & Flood Early-Warning Model</h1>", unsafe_allow_html=True)
st.caption("Live rainfall-to-flood prediction — Team Greedy Algo")

mode = st.sidebar.radio("Mode", ["Replay real event", "Custom rainfall (what-if)"])

if mode == "Replay real event":
    # known validation events from evaluate.py's table — guaranteed to exist in the cache
    event_id = st.sidebar.selectbox("Storm event", [21, 66, 4, 50, 60, 59, 18, 17, 7, 61, 33, 42, 67])
    depth_full = compute_replay_rollout(event_id)   # instant after the first run for this event
    T = depth_full.shape[0]
    t = st.sidebar.slider("Timestep", 0, T - 1, WARMUP)
    depth_now = depth_full[t]
    rain_now = cache[f"rain_{event_id}"][t]
    stage_label = "warmup (ground truth)" if t < WARMUP else "rollout (model prediction)"
else:
    st.sidebar.caption(f"Training data's peak rainfall was {all_rain_max:.3f} per step — values well above this are extrapolation, not reliable prediction.")
    custom_rain = st.sidebar.number_input(
        "Rainfall (mm, per step)", min_value=0.0, max_value=float(all_rain_max * 3),
        value=float(all_rain_max * 0.5), step=float(all_rain_max / 20), format="%.3f",
    )
    if custom_rain > all_rain_max:
        st.sidebar.warning("Above the max seen in training — the model is extrapolating here, treat the output as unreliable.")
    depth_full = compute_custom_rollout(custom_rain)   # instant after the first run for this rainfall value
    t = st.sidebar.slider("Timestep", 0, MAX_CUSTOM_STEPS - 1, 20)
    depth_now = depth_full[t]
    rain_now = custom_rain
    stage_label = "model prediction (no ground truth — hypothetical)"

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0E1420')
    ax.set_facecolor('#0E1420')
    vmax = max(float(depth_full.max()), 0.1)   # fixed across the whole scenario so the colorbar doesn't jump around as you drag
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=depth_now, cmap="Blues", vmin=0, vmax=vmax, s=8)
    ax.set_title(f"t = {t}  ({stage_label})", color='#E7ECF3')
    ax.set_aspect("equal")
    ax.tick_params(colors='#8291AA')
    for spine in ax.spines.values():
        spine.set_color('#24304A')
    cbar = plt.colorbar(sc, label="Predicted depth (m)")
    cbar.ax.yaxis.label.set_color('#E7ECF3')
    cbar.ax.tick_params(colors='#8291AA')
    st.pyplot(fig)

with col2:
    st.metric("Max predicted depth", f"{depth_now.max():.2f} m")
    st.metric("Nodes flooded (>0.1m)", int((depth_now > 0.1).sum()))
    st.metric("Current rainfall input", f"{rain_now:.2f} mm")