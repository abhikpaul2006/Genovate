import numpy as np, torch
from model import FloodGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

WARMUP = 10  # matches the assumed test-event format — first WARMUP steps given, unscored

g = np.load("graph_model1.npz")
node_static = torch.tensor(g["node_features"], dtype=torch.float32, device=device)
src = torch.tensor(g["edge_index"][0], dtype=torch.long, device=device)
dst = torch.tensor(g["edge_index"][1], dtype=torch.long, device=device)
N1D, N2D = int(g["N1D"]), int(g["N2D"])
N = N1D + N2D

edge_type_oh = np.eye(3, dtype="float32")[g["edge_type"].astype("int64")]
efr = g["edge_feat_raw"].astype("float32")
efr = (efr - efr.mean(0)) / (efr.std(0) + 1e-6)
edge_feat = torch.tensor(np.concatenate([efr, edge_type_oh], axis=1), dtype=torch.float32, device=device)

cache = np.load("cache_model1.npz")
events = list(cache["events"])
mask2d = cache["mask"]

rng = np.random.default_rng(42)   # same seed/split as train.py — val_idx must match
idx = rng.permutation(len(events))
n_val = max(1, len(events) // 5)
val_idx, tr_idx = idx[:n_val], idx[n_val:]

all_rain_max, all_cum_max = 0.0, 0.0
for i in range(len(events)):
    r = cache[f"rain_{i}"]
    all_rain_max = max(all_rain_max, float(r.max()))
    all_cum_max = max(all_cum_max, float(np.cumsum(r).max()))
all_rain_max = max(all_rain_max, 1e-6)
all_cum_max = max(all_cum_max, 1e-6)

def transform(d): return np.sqrt(np.clip(d, 0, None))
def inv_transform(d): return d ** 2

def event_arrays(i):
    depth2d = np.nan_to_num(cache[f"depth_{i}"], nan=0.0).astype("float32")
    rain = cache[f"rain_{i}"].astype("float32")
    return depth2d, rain

model = FloodGNN(node_in=13 + 4, edge_in=edge_feat.shape[1], hidden=64, layers=4).to(device)
model.load_state_dict(torch.load("flood_gnn_model1.pt", map_location=device))
model.eval()

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
    return pred[0].detach().cpu().numpy()

def eval_full_teacher_forced(i):
    depth2d, rain = event_arrays(i)
    T = depth2d.shape[0]
    cum = np.cumsum(rain)
    se, n = 0.0, 0
    for t in range(T):
        prevd = transform(depth2d[t - 1]) if t > 0 else np.zeros(N2D, dtype="float32")
        pred = predict_step(prevd, rain[t], cum[t], t / max(T - 1, 1))
        p = inv_transform(np.clip(pred[N1D:][mask2d], 0, None))
        t_ = inv_transform(transform(depth2d[t])[mask2d])
        se += np.sum((p - t_) ** 2); n += p.size
    return (se / n) ** 0.5

def eval_rollout(i, warmup=WARMUP):
    """FIXED: was `if t < warmup`, which fed the model's own (already-imperfect)
    prediction from t-1 as input at the very first scored step, instead of the
    true, known depth at t-1. Should be `t <= warmup` — the previous timestep
    is only unknown once t-1 itself is >= warmup."""
    depth2d, rain = event_arrays(i)
    T = depth2d.shape[0]
    cum = np.cumsum(rain)
    se, n = 0.0, 0
    last_pred_2d = None
    for t in range(T):
        if t <= warmup:
            prevd = transform(depth2d[t - 1]) if t > 0 else np.zeros(N2D, dtype="float32")
        else:
            prevd = last_pred_2d
        pred = predict_step(prevd, rain[t], cum[t], t / max(T - 1, 1))
        pred_2d = np.clip(pred[N1D:], 0, None)
        last_pred_2d = pred_2d
        if t >= warmup:
            p = inv_transform(pred_2d[mask2d])
            t_ = inv_transform(transform(depth2d[t])[mask2d])
            se += np.sum((p - t_) ** 2); n += p.size
    return (se / n) ** 0.5 if n > 0 else float("nan")

print(f"\n{'event':>6} | {'full (teacher-forced)':>22} | {'rollout (warmup='+str(WARMUP)+')':>22}")
full_scores, roll_scores = [], []
for i in val_idx:
    i = int(i)
    fscore = eval_full_teacher_forced(i)
    rscore = eval_rollout(i)
    full_scores.append(fscore); roll_scores.append(rscore)
    print(f"{i:6d} | {fscore:22.4f} | {rscore:22.4f}")

print(f"\nmean full (teacher-forced) RMSE: {np.mean(full_scores):.4f}")
print(f"mean rollout RMSE:               {np.mean(roll_scores):.4f}")
print(f"baseline (per-node-mean) RMSE:    0.328")