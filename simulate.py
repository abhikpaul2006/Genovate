"""
simulate.py — animate predicted flood depth over the real drainage network
for one validation event, using real node positions from the static CSV.

Produces flood_simulation.gif in the same folder.

Run from your `flood prediction` project folder, next to
model.py, graph_model1.npz, cache_model1.npz, flood_gnn_model1.pt.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from model import FloodGNN

# ----------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = r"D:\Dataset_Rerelease\Models\Model_1\train"   # same root graph.py used, for static positions
EVENT_ID = 17     # index into cache_model1.npz's events (matches evaluate.py's table: event 17)
WARMUP = 10       # matches evaluate.py

def transform(d): return np.sqrt(np.clip(d, 0, None))
def inv_transform(d): return d ** 2

# ----------------------------------------------------------------------
# 2. Load static graph + trained model
# ----------------------------------------------------------------------
g = np.load("graph_model1.npz")
node_static = torch.tensor(g["node_features"], dtype=torch.float32, device=device)
src = torch.tensor(g["edge_index"][0], dtype=torch.long, device=device)
dst = torch.tensor(g["edge_index"][1], dtype=torch.long, device=device)
N1D, N2D = int(g["N1D"]), int(g["N2D"])
N = N1D + N2D

# edge_feat has to be rebuilt the same way train.py / evaluate.py build it —
# graph_model1.npz only stores the raw pieces, not the combined tensor.
edge_type_oh = np.eye(3, dtype="float32")[g["edge_type"].astype("int64")]
efr = g["edge_feat_raw"].astype("float32")
efr = (efr - efr.mean(0)) / (efr.std(0) + 1e-6)
edge_feat = torch.tensor(np.concatenate([efr, edge_type_oh], axis=1), dtype=torch.float32, device=device)

model = FloodGNN(node_in=13 + 4, edge_in=edge_feat.shape[1], hidden=64, layers=4).to(device)
model.load_state_dict(torch.load("flood_gnn_model1.pt", map_location=device))
model.eval()

# ----------------------------------------------------------------------
# 3. Real node positions — from the same CSV + sort order graph.py used,
#    so rows line up with node_features[N1D:] with no alignment risk.
# ----------------------------------------------------------------------
n2d = pd.read_csv(os.path.join(ROOT, "2d_nodes_static.csv")).sort_values("node_idx").reset_index(drop=True)
xy = n2d[["position_x", "position_y"]].values.astype("float32")
assert xy.shape[0] == N2D, f"position rows ({xy.shape[0]}) != N2D ({N2D}) — sort order mismatch"

# ----------------------------------------------------------------------
# 4. Load one event's rainfall + true depth
# ----------------------------------------------------------------------
cache = np.load("cache_model1.npz")
depth2d = np.nan_to_num(cache[f"depth_{EVENT_ID}"], nan=0.0).astype("float32")   # (T, N2D), raw depth
rain = cache[f"rain_{EVENT_ID}"].astype("float32")
T = depth2d.shape[0]
cum = np.cumsum(rain)

# same global scale train.py/evaluate.py compute — recomputed here instead of
# hardcoded, so it can't silently go stale if the cache is ever regenerated
all_events = list(cache["events"])
all_rain_max, all_cum_max = 0.0, 0.0
for i in range(len(all_events)):
    r = cache[f"rain_{i}"]
    all_rain_max = max(all_rain_max, float(r.max()))
    all_cum_max = max(all_cum_max, float(np.cumsum(r).max()))
all_rain_max = max(all_rain_max, 1e-6)
all_cum_max = max(all_cum_max, 1e-6)

# ----------------------------------------------------------------------
# 5. Rollout: steps < WARMUP shown as ground truth (no model needed);
#    steps >= WARMUP predicted, first one seeded from the true depth just
#    before it, every one after that fed by the model's own last guess.
# ----------------------------------------------------------------------
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

depth_display = np.zeros((T, N2D), dtype="float32")
depth_display[:WARMUP] = depth2d[:WARMUP]   # ground truth for the given steps, no model call needed

last_pred_2d = None  # sqrt-depth, the model's own last guess
for t in range(WARMUP, T):
    if t <= WARMUP:
        prevd = transform(depth2d[t - 1])   # last known TRUE value — fixes the off-by-one
    else:
        prevd = last_pred_2d                 # model's own last guess, no more truth available
    pred = predict_step(prevd, rain[t], cum[t], t / max(T - 1, 1))
    pred_2d_sqrt = np.clip(pred[N1D:], 0, None)
    last_pred_2d = pred_2d_sqrt
    depth_display[t] = inv_transform(pred_2d_sqrt)

# ----------------------------------------------------------------------
# 6. Animate
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
vmax = np.percentile(depth_display, 99)
sc = ax.scatter(xy[:, 0], xy[:, 1], c=depth_display[0], cmap="Blues", vmin=0, vmax=vmax, s=6)
title = ax.set_title(f"Event {EVENT_ID} — t=0")
ax.set_aspect("equal")
plt.colorbar(sc, label="Predicted depth (m)")

def update(frame):
    sc.set_array(depth_display[frame])
    title.set_text(f"Event {EVENT_ID} — t={frame} {'(warmup)' if frame < WARMUP else '(rollout)'}")
    return sc, title

ani = animation.FuncAnimation(fig, update, frames=T, interval=150, blit=False)
ani.save("flood_simulation.gif", writer="pillow", fps=6)
print("Saved flood_simulation.gif")