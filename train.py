import numpy as np, torch, torch.nn as nn
from model import FloodGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

BATCH_T = 24              # length of each training mini-rollout window
STEPS_PER_EPOCH = 35   # halved vs. before — each "step" now does BATCH_T sequential forward/backward passes, not 1 batched one
TOTAL_EPOCHS = 30        # more room for the sampling-probability ramp than the old 15
FULL_EVAL_EVERY = 5      # run the full, all-events rollout eval this often; quick (subset) eval every other epoch
QUICK_EVAL_EVENTS = 4    # how many val events to check on non-full-eval epochs, for a fast per-epoch signal
QUICK_EVAL_MAX_STEPS = 60
WARMUP = 10              # matches evaluate.py / the assumed test-event format
MAX_SAMPLING_PROB = 0.5  # cap on how often training uses the model's own prediction instead of ground truth

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
mask_full = torch.zeros(N, dtype=torch.bool, device=device)
mask_full[N1D:] = torch.tensor(mask2d, device=device)

rng = np.random.default_rng(42)
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
print("rain max:", all_rain_max, "| cum max:", all_cum_max)

def transform(d): return np.sqrt(np.clip(d, 0, None))
def inv_transform(d): return d ** 2

def event_arrays(i):
    depth2d = np.nan_to_num(cache[f"depth_{i}"], nan=0.0).astype("float32")
    rain = cache[f"rain_{i}"].astype("float32")
    return depth2d, rain
all_depth_max_sqrt = 0.0
for i in range(len(events)):
    d2d, _ = event_arrays(i)
    all_depth_max_sqrt = max(all_depth_max_sqrt, float(transform(d2d).max()))
PRED_CAP = all_depth_max_sqrt * 1.5
print("pred cap (sqrt-depth):", PRED_CAP)
model = FloodGNN(node_in=13 + 4, edge_in=edge_feat.shape[1], hidden=64, layers=4).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

def forward_single(prev_depth_transformed_t, rain_t, cum_t, t_frac, grad):
    """One timestep, one node graph. prev_depth_transformed_t can be a torch tensor
    (grad=True path, keeps autograd history) or a numpy array (grad=False path)."""
    dyn = torch.zeros((1, N, 4), dtype=torch.float32, device=device)
    if torch.is_tensor(prev_depth_transformed_t):
        dyn[0, N1D:, 0] = prev_depth_transformed_t
    else:
        dyn[0, N1D:, 0] = torch.tensor(prev_depth_transformed_t, dtype=torch.float32, device=device)
    dyn[0, :, 1] = float(rain_t) / all_rain_max
    dyn[0, :, 2] = float(cum_t) / all_cum_max
    dyn[0, :, 3] = float(t_frac)
    static = node_static.unsqueeze(0)
    x = torch.cat([static, dyn], dim=-1)
    with torch.set_grad_enabled(grad):
        pred = model(x, edge_feat, src, dst)
    return pred[0]  # (N,), still on device, still attached to graph if grad=True

def sampling_prob(epoch):
    # linear ramp: 0 at epoch 0, MAX_SAMPLING_PROB by TOTAL_EPOCHS
    return MAX_SAMPLING_PROB * min(1.0, epoch / max(TOTAL_EPOCHS - 1, 1))

def train_step(i, p_self):
    """Trains on one real BATCH_T-length mini-rollout from event i.
    At each internal step, with probability p_self the model's own last
    prediction (detached) is fed in as 'previous depth' instead of the
    true value — this is what actually teaches it to recover from its
    own errors, which teacher forcing alone never does."""
    depth2d, rain = event_arrays(i)
    T = depth2d.shape[0]
    start = int(rng.integers(0, max(T - BATCH_T, 1)))
    end = min(start + BATCH_T, T)
    cum = np.cumsum(rain)

    prev_input = np.zeros(N2D, dtype="float32") if start == 0 else transform(depth2d[start - 1])

    model.train()
    opt.zero_grad()
    total_loss = 0.0
    for t in range(start, end):
        pred = forward_single(prev_input, rain[t], cum[t], t / max(T - 1, 1), grad=True)
        tgt = torch.zeros(N, dtype=torch.float32, device=device)
        tgt[N1D:] = torch.tensor(transform(depth2d[t]), dtype=torch.float32, device=device)
        diff = (pred - tgt)[mask_full]
        total_loss = total_loss + (diff ** 2).mean()

        pred_2d = np.clip(pred[N1D:].detach().cpu().numpy(), 0, PRED_CAP)   # <-- changed from None
        if rng.random() < p_self:
            prev_input = pred_2d                       # self-fed: model sees its own (imperfect) output
        else:
            prev_input = transform(depth2d[t])          # teacher-forced: true value, as before

    total_loss = total_loss / (end - start)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return total_loss.item()

def eval_rollout(i, warmup=WARMUP, max_steps=None):
    """Full rollout metric: true depth fed for the first `warmup` steps
    (unscored), the model's own prediction fed after that, scored from
    `warmup` onward. This is the number that matters — the one that
    matches what the real test events require."""
    depth2d, rain = event_arrays(i)
    T = depth2d.shape[0]
    cum = np.cumsum(rain)
    steps = np.arange(T) if (max_steps is None or T <= max_steps) else \
        np.unique(np.linspace(warmup, T - 1, max_steps).astype(int))

    model.eval()
    last_pred = PRED_CAP
    se, n = 0.0, 0
    with torch.no_grad():
        for t in range(T):
            if t <= warmup:
                prevd = transform(depth2d[t - 1]) if t > 0 else np.zeros(N2D, dtype="float32")
            else:
                prevd = last_pred
            pred = forward_single(prevd, rain[t], cum[t], t / max(T - 1, 1), grad=False)
            pred_2d = np.clip(pred[N1D:].detach().cpu().numpy(), 0, None)
            last_pred = pred_2d
            if t >= warmup and t in steps:
                p = inv_transform(pred_2d[mask2d])
                t_ = inv_transform(transform(depth2d[t])[mask2d])
                se += np.sum((p - t_) ** 2); n += p.size
    return (se / n) ** 0.5 if n > 0 else float("nan")

best_rmse = float("inf")
for epoch in range(TOTAL_EPOCHS):
    p_self = sampling_prob(epoch)
    tr_losses = [train_step(int(rng.choice(tr_idx)), p_self) for _ in range(STEPS_PER_EPOCH)]

    do_full = (epoch + 1) % FULL_EVAL_EVERY == 0
    eval_events = val_idx if do_full else val_idx[:QUICK_EVAL_EVENTS]
    max_steps = None if do_full else QUICK_EVAL_MAX_STEPS
    val_rmses = [eval_rollout(int(i), max_steps=max_steps) for i in eval_events]
    mean_rmse = float(np.mean(val_rmses))
    tag = "full" if do_full else "quick"

    print(f"epoch {epoch+1:2d} | p_self {p_self:.2f} | train loss {np.mean(tr_losses):.4f} | rollout RMSE({tag}) {mean_rmse:.4f}")

    if do_full and mean_rmse < best_rmse:
        best_rmse = mean_rmse
        torch.save(model.state_dict(), "flood_gnn_model1.pt")
        print(f"  -> new best full rollout RMSE ({best_rmse:.4f}), saved flood_gnn_model1.pt")

print(f"done. best full rollout RMSE seen: {best_rmse:.4f} (baseline: 0.328)")