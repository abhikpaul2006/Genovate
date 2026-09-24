import numpy as np, pandas as pd, os

root = r"D:\Dataset_Rerelease\Models\Model_1\train"
n1d = pd.read_csv(os.path.join(root, "1d_nodes_static.csv")).sort_values("node_idx").reset_index(drop=True)
n2d = pd.read_csv(os.path.join(root, "2d_nodes_static.csv")).sort_values("node_idx").reset_index(drop=True)
e1d = pd.read_csv(os.path.join(root, "1d_edges_static.csv"))
ei1d = pd.read_csv(os.path.join(root, "1d_edge_index.csv"))
e2d = pd.read_csv(os.path.join(root, "2d_edges_static.csv"))
ei2d = pd.read_csv(os.path.join(root, "2d_edge_index.csv"))
conn = pd.read_csv(os.path.join(root, "1d2d_connections.csv"))

N1D, N2D = len(n1d), len(n2d)
N = N1D + N2D
print("nodes: 1d", N1D, "| 2d", N2D, "| total", N)

cache = np.load("cache_model1.npz")
is_boundary_2d = ~cache["mask"]

# --- diagnostic: which n2d columns are NaN for boundary rows? ---
print("NaN counts per n2d column (boundary rows only):")
print(n2d.loc[is_boundary_2d].isna().sum())

def z(x):
    x = np.asarray(x, dtype="float32")
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6)

is_1d = np.zeros(N, "float32"); is_1d[:N1D] = 1
is_boundary = np.zeros(N, "float32"); is_boundary[N1D:] = is_boundary_2d
pos_x = np.concatenate([n1d.position_x.values, n2d.position_x.values]).astype("float32")
pos_y = np.concatenate([n1d.position_y.values, n2d.position_y.values]).astype("float32")
min_elev = np.concatenate([n1d.invert_elevation.values, n2d.min_elevation.values]).astype("float32")
surf_elev = np.concatenate([n1d.surface_elevation.values, n2d.elevation.values]).astype("float32")
area = np.concatenate([n1d.base_area.values, n2d.area.values]).astype("float32")
pipe_depth = np.zeros(N, "float32"); pipe_depth[:N1D] = n1d.depth.values
roughness = np.zeros(N, "float32"); roughness[N1D:] = n2d.roughness.values
aspect = np.zeros(N, "float32"); aspect[N1D:] = n2d.aspect.values
curvature = np.zeros(N, "float32"); curvature[N1D:] = n2d.curvature.values
flow_acc = np.zeros(N, "float32"); flow_acc[N1D:] = np.log1p(n2d.flow_accumulation.values)

node_features = np.stack([is_1d, 1 - is_1d, is_boundary, z(pos_x), z(pos_y),
    z(min_elev), z(surf_elev), z(area), z(pipe_depth), z(roughness),
    z(aspect), z(curvature), z(flow_acc)], axis=1)
print("node_features shape:", node_features.shape)

n_nan = np.isnan(node_features).sum()
print(f"node_features: {n_nan} NaN values found" + (" (zeroing)" if n_nan else ""))
node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

f2, t2 = ei2d.from_node.values + N1D, ei2d.to_node.values + N1D
length2, slope2, face2 = e2d.length.values.astype("float32"), e2d.slope.values.astype("float32"), e2d.face_length.values.astype("float32")
ef2f = np.stack([length2, slope2, face2, np.zeros_like(length2)], axis=1)
ef2r = np.stack([length2, -slope2, face2, np.zeros_like(length2)], axis=1)

f1, t1 = ei1d.from_node.values, ei1d.to_node.values
length1, slope1, diam1 = e1d.length.values.astype("float32"), e1d.slope.values.astype("float32"), e1d.diameter.values.astype("float32")
ef1f = np.stack([length1, slope1, np.zeros_like(length1), diam1], axis=1)
ef1r = np.stack([length1, -slope1, np.zeros_like(length1), diam1], axis=1)

fc, tc = conn.node_1d.values, conn.node_2d.values + N1D
lenc = np.sqrt((pos_x[tc]-pos_x[fc])**2 + (pos_y[tc]-pos_y[fc])**2).astype("float32")
efc = np.stack([lenc, np.zeros_like(lenc), np.zeros_like(lenc), np.zeros_like(lenc)], axis=1)

edge_index = np.concatenate([
    np.stack([f2, t2]), np.stack([t2, f2]),
    np.stack([f1, t1]), np.stack([t1, f1]),
    np.stack([fc, tc]), np.stack([tc, fc]),
], axis=1)
edge_feat_raw = np.concatenate([ef2f, ef2r, ef1f, ef1r, efc, efc], axis=0)
edge_type = np.concatenate([np.zeros(len(f2)*2), np.ones(len(f1)*2), np.full(len(fc)*2, 2)]).astype("float32")

n_nan_e = np.isnan(edge_feat_raw).sum()
print(f"edge_feat_raw: {n_nan_e} NaN values found" + (" (zeroing)" if n_nan_e else ""))
edge_feat_raw = np.nan_to_num(edge_feat_raw, nan=0.0, posinf=0.0, neginf=0.0)

print("edge_index shape:", edge_index.shape)
print("edge_type counts:", np.unique(edge_type, return_counts=True))

assert not np.isnan(node_features).any(), "NaNs still in node_features!"
assert not np.isnan(edge_feat_raw).any(), "NaNs still in edge_feat_raw!"

np.savez_compressed("graph_model1.npz", node_features=node_features, edge_index=edge_index,
    edge_feat_raw=edge_feat_raw, edge_type=edge_type, is_boundary=is_boundary, N1D=N1D, N2D=N2D)
print("saved graph_model1.npz")