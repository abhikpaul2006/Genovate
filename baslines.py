import numpy as np

data = np.load("cache_model1.npz")
events = data["events"]
mask = data["mask"]  # True = real node, False = boundary
depths = [data[f"depth_{i}"] for i in range(len(events))]

print("num events:", len(depths))
print("lengths range:", min(len(d) for d in depths), "-", max(len(d) for d in depths))

rng = np.random.default_rng(42)
idx = rng.permutation(len(depths))
n_val = max(1, len(depths) // 5)
val_idx, tr_idx = idx[:n_val], idx[n_val:]

train_stack = np.concatenate([depths[i] for i in tr_idx], axis=0)
mean_depth = np.nanmean(train_stack[:, mask], axis=0)

def rmse(pred_fn):
    se, n = 0.0, 0
    for i in val_idx:
        d = depths[i][:, mask]
        p = pred_fn(d)
        diff = d - p
        se += np.nansum(diff ** 2)
        n += np.sum(~np.isnan(diff))
    return (se / n) ** 0.5

print("train events:", len(tr_idx), "| val events:", len(val_idx))
print("RMSE predict 0     :", rmse(lambda d: 0))
print("RMSE per-node mean :", rmse(lambda d: mean_depth))