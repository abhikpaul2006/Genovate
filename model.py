import torch
import torch.nn as nn

def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class FloodGNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden=64, layers=4):
        super().__init__()
        self.node_enc = mlp([node_in, hidden, hidden])
        self.edge_enc = mlp([edge_in, hidden, hidden])
        self.msg_mlps = nn.ModuleList([mlp([3 * hidden, hidden, hidden]) for _ in range(layers)])
        self.upd_mlps = nn.ModuleList([mlp([2 * hidden, hidden, hidden]) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.decoder = mlp([hidden, hidden, 1])
        self.layers = layers

    def forward(self, node_x, edge_x, src, dst):
        # node_x: [T, N, node_in]  edge_x: [E, edge_in]  src, dst: [E]
        T, N, _ = node_x.shape
        h = self.node_enc(node_x)
        e = self.edge_enc(edge_x).unsqueeze(0).expand(T, -1, -1)
        for k in range(self.layers):
            msg_in = torch.cat([h[:, src, :], h[:, dst, :], e], dim=-1)
            msg = self.msg_mlps[k](msg_in)
            agg = torch.zeros_like(h)
            agg.index_add_(1, dst, msg)
            h = self.norms[k](h + self.upd_mlps[k](torch.cat([h, agg], dim=-1)))
        return self.decoder(h).squeeze(-1)  # [T, N]