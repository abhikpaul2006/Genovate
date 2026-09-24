import os, numpy as np, pandas as pd

root = r"D:\Dataset_Rerelease\Models\Model_1\test"
st2d = pd.read_csv(os.path.join(root, "2d_nodes_static.csv")).sort_values("node_idx")
minel = st2d.min_elevation.values.astype("float32")
N2D = len(st2d)

events = sorted([d for d in os.listdir(root) if d.startswith("event_")],
                key=lambda s: int(s.split("_")[1]))
print("num test events found:", len(events))
print("event ids:", events)

depths, rains, lens = [], [], []
for e in events:
    df = pd.read_csv(os.path.join(root, e, "2d_nodes_dynamic_all.csv"),
                      usecols=["timestep", "node_idx", "rainfall", "water_level"])
    wl = df.pivot(index="timestep", columns="node_idx", values="water_level").values.astype("float32")
    rn = df.pivot(index="timestep", columns="node_idx", values="rainfall").values.astype("float32")
    depths.append(wl - minel)
    rains.append(rn.mean(axis=1))
    lens.append(wl.shape[0])

print("timestep counts:", sorted(set(lens)))
mask = ~np.isnan(depths[0]).all(axis=0)
print("non-boundary nodes:", mask.sum(), "/", N2D)

np.savez_compressed(
    "cache_model1_test.npz",
    node_idx=st2d.node_idx.values,
    mask=mask,
    min_elevation=minel,
    events=np.array(events),
    **{f"depth_{i}": d for i, d in enumerate(depths)},
    **{f"rain_{i}": r for i, r in enumerate(rains)},
)
print("saved cache_model1_test.npz")