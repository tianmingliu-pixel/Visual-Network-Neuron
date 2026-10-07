"""
扩展模板 / Extension template
复制本文件并去掉文件名开头的下划线（例如 gnn.py），即可注册新网络。
Copy this file, drop the leading underscore (e.g. gnn.py), and edit.

示例：一个最小的图神经网络层（GCN），作为未来 "graph" 领域的起点。
Example: a minimal GCN as a starting point for the future "graph" domain.
"""
import torch
import torch.nn as nn

from neurocore.registry import register_model


class GCNLayer(nn.Module):
    """H' = σ( D^-1/2 (A+I) D^-1/2 · H · W )"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h, adj):                   # h: (N, F), adj: (N, N) 0/1
        a = adj + torch.eye(adj.size(0), device=adj.device)
        d = a.sum(-1).pow(-0.5)
        a = d[:, None] * a * d[None, :]
        return a @ self.lin(h)


@register_model("gcn_example", domain="graph", description="Minimal 2-layer GCN (extension example)")
class GCN(nn.Module):
    def __init__(self, in_dim=16, hidden=32, num_classes=4):
        super().__init__()
        self.l1, self.l2 = GCNLayer(in_dim, hidden), GCNLayer(hidden, num_classes)

    def forward(self, h, adj):
        return self.l2(torch.relu(self.l1(h, adj)), adj)
