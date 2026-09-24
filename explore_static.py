import pandas as pd, os

root = r"D:\Dataset_Rerelease\Models\Model_1\train"
files = [
    "1d_nodes_static.csv", "1d_edges_static.csv", "1d_edge_index.csv",
    "2d_nodes_static.csv", "2d_edges_static.csv", "2d_edge_index.csv",
    "1d2d_connections.csv", "dataset_summary.csv",
]
for f in files:
    df = pd.read_csv(os.path.join(root, f))
    print("\n==", f, "| shape:", df.shape)
    print(list(df.columns))
    print(df.head(2).to_string())