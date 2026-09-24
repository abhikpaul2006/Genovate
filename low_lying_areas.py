import numpy as np, pandas as pd, os

ROOT = r"D:\Dataset_Rerelease\Models\Model_1\train"

cache = np.load("cache_model1.npz")
events = list(cache["events"])
mask = cache["mask"]  # True = real node, False = boundary (no usable depth)

# stack depth across every event and every timestep -> one long (T_total, N2D) array
all_depths = np.concatenate([cache[f"depth_{i}"] for i in range(len(events))], axis=0)
mean_depth = np.nanmean(all_depths, axis=0)   # how deep it floods ON AVERAGE
max_depth = np.nanmax(all_depths, axis=0)     # worst-case depth ever seen

n2d = pd.read_csv(os.path.join(ROOT, "2d_nodes_static.csv")).sort_values("node_idx").reset_index(drop=True)

df = pd.DataFrame({
    "node_idx": n2d.node_idx.values,
    "x": n2d.position_x.values,
    "y": n2d.position_y.values,
    "elevation": n2d.min_elevation.values,
    "flow_accumulation": n2d.flow_accumulation.values,
    "mean_depth": mean_depth,
    "max_depth": max_depth,
})
df = df[mask].reset_index(drop=True)  # drop boundary nodes, they have no real depth signal

low_lying = df.sort_values("mean_depth", ascending=False).head(20)
print("Top 20 most flood-prone real points (by average depth across all events):\n")
print(low_lying[["node_idx", "x", "y", "elevation", "mean_depth", "max_depth"]].to_string(index=False))

df.sort_values("mean_depth", ascending=False).to_csv("low_lying_areas.csv", index=False)
print("\nSaved full ranked list to low_lying_areas.csv")