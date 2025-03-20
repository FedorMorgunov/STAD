import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """
    A simple Graph Attention Layer that works on a batch of graphs.
    Assumes that each graph has a fixed number of nodes (joints)
    and that an adjacency matrix is provided.
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat  # if True, apply nonlinearity after output

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism: a single weight vector to compute attention coefficients
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        Args:
            h: Tensor of shape (B, N, in_features) – input features for N nodes in each of B graphs.
            adj: Tensor of shape (B, N, N) – binary (or weighted) adjacency matrices.
        Returns:
            h_prime: Tensor of shape (B, N, out_features)
        """
        B, N, _ = h.size()
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (B, N, out_features)

        # Prepare attention input by concatenating each pair of node features
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (B, N, N, out_features)
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, N, out_features)
        e = torch.cat([Wh1, Wh2], dim=-1)  # (B, N, N, 2*out_features)
        e = self.leakyrelu(torch.matmul(e, self.a).squeeze(-1))  # (B, N, N)

        # Mask attention values where there is no edge.
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # (B, N, out_features)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class AdvancedGraphEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, dropout=0.1, alpha=0.2):
        """
        Args:
            in_features: input feature dimension (e.g., 2 for (x,y))
            hidden_features: intermediate dimension (e.g., 64)
            out_features: final output dimension (e.g., 128)
            num_layers: total number of GAT layers to stack
            dropout: dropout probability for attention weights
            alpha: negative slope for LeakyReLU
        """
        super(AdvancedGraphEncoder, self).__init__()
        layers = []
        # First layer: from input to hidden
        layers.append(GATLayer(in_features, hidden_features, dropout=dropout, alpha=alpha, concat=True))
        # Intermediate layers (if any)
        for _ in range(num_layers - 2):
            layers.append(GATLayer(hidden_features, hidden_features, dropout=dropout, alpha=alpha, concat=True))
        # Final layer: from hidden to output; no concatenation for clean output
        layers.append(GATLayer(hidden_features, out_features, dropout=dropout, alpha=alpha, concat=False))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, h, adj):
        for layer in self.layers:
            h = layer(h, adj)
        return h
