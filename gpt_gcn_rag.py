# =====================================================
# Refactored from gpt_gat.py:
#   - GATConv -> GCNConv
# --- MLP-based nn.Sequential classifier
#  
#   - RAG (GraphMemory + RAGRetriever + Fusion) incorporated
#     as in Revised_RAG_ETGN.py
# =====================================================

# =====================================================
# Imports
# =====================================================
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import GPT2Model, GPT2Config
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns


# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# MGGPT: GCN + GPT-2 backbone
#
# =====================================================
class MGGPT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=2, dropout=0.1):
        super(MGGPT, self).__init__()
        self.hidden_dim = hidden_dim

        # GNN layers -- GCNConv replaces GATConv.
        # GCNConv has no `heads` argument, so output channels stay = hidden_dim,
        # which now matches GPT-2's n_embd cleanly.
        self.gcn_input = GCNConv(input_dim, hidden_dim)
        self.gcn_hidden = GCNConv(hidden_dim, hidden_dim)

        # GPT-2 configuration
        self.gpt_config = GPT2Config(
            vocab_size=2,
            n_embd=hidden_dim,
            n_layer=2,
            n_head=num_heads
        )
        self.gpt = GPT2Model(self.gpt_config)

        # Linear layer for combining GNN and GPT outputs
        self.linear_combine = nn.Linear(hidden_dim * 2, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, h_c, batch=None):
        # ---- GNN part (missing-info module removed) ----
        x_gnn = F.relu(self.gcn_input(x, edge_index))
        x_gnn = self.dropout(x_gnn)
        x_gnn = F.relu(self.gcn_hidden(x_gnn, edge_index))

        # ---- GPT part ----
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()
        padded_x = torch.zeros(
            (num_graphs, max_nodes, self.hidden_dim), device=x.device
        )

        for i in range(num_graphs):
            node_indices = (batch == i).nonzero(as_tuple=True)[0]
            padded_x[i, :len(node_indices), :] = x_gnn[node_indices]

        gpt_output = self.gpt(inputs_embeds=padded_x).last_hidden_state
        x_gpt_flat = torch.cat(
            [gpt_output[i, :torch.sum(batch == i)] for i in range(num_graphs)],
            dim=0
        )

        if x_gpt_flat.shape[0] != x_gnn.shape[0]:
            raise ValueError(
                f"Shape mismatch: x_gnn={x_gnn.shape}, x_gpt={x_gpt_flat.shape}"
            )

        combined = torch.cat([x_gnn, x_gpt_flat], dim=-1)
        output = self.linear_combine(combined)
        output = self.dropout(output)

        if h_c is None:
            h = output
            c = torch.zeros_like(h)
        else:
            h, c = h_c
            h = output

        return h, c


# =====================================================
# MLP-based EdgeClassifier
#
# =====================================================
class EdgeClassifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# RAG components
# =====================================================
class GraphMemory:
    def __init__(self, max_size=5000):
        self.embeddings = []
        self.labels = []
        self.max_size = max_size

    def add(self, e, y):
        self.embeddings.append(e.detach().cpu())
        self.labels.append(y.detach().cpu())
        if len(self.embeddings) > self.max_size:
            self.embeddings.pop(0)
            self.labels.pop(0)

    def get(self):
        if not self.embeddings:
            return None, None
        return torch.cat(self.embeddings), torch.cat(self.labels)


class RAGRetriever(nn.Module):
    def __init__(self, dim, k=5):
        super().__init__()
        self.k = k
        self.scale = dim ** -0.5

    def forward(self, q, mem):
        sim = self.scale * torch.matmul(q, mem.T)
        # Cold-start guard: memory may have fewer than k entries early on.
        k = min(self.k, mem.size(0))
        idx = torch.topk(sim, k, dim=1).indices
        return mem[idx].mean(dim=1)


class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, x, c):
        return F.relu(self.fc(torch.cat([x, c], dim=1)))


# =====================================================
# Enhanced Temporal Graph Network (with RAG)
# =====================================================
class EnhancedTemporalGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(EnhancedTemporalGraphNetwork, self).__init__()
        self.mggpt = MGGPT(input_dim, hidden_dim)
        self.num_layers = num_layers

        # Edge embedding = [h_src ; h_dst] -> dim = hidden_dim * 2
        edge_dim = hidden_dim * 2

        # RAG modules
        self.memory = GraphMemory()
        self.retriever = RAGRetriever(edge_dim)
        self.fusion = Fusion(edge_dim)

        # MLP classifier
        self.classifier = EdgeClassifier(edge_dim)

    def create_edge_embeddings(self, node_embeddings, edge_index):
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)
        return edge_embeddings

    def forward(self, x, edge_index, batch=None, edge_labels=None, training=True):
        h_c = None
        for _ in range(self.num_layers):
            h, c = self.mggpt(x, edge_index, h_c, batch)
            h_c = (h, c)

        edge_emb = self.create_edge_embeddings(h, edge_index)

        # ---- RAG: retrieve historical context and fuse ----
        mem, _ = self.memory.get()
        if mem is not None:
            ctx = self.retriever(edge_emb, mem.to(edge_emb.device))
            edge_emb = self.fusion(edge_emb, ctx)

        logits = self.classifier(edge_emb).squeeze(-1)

        # Write to memory during training
        if training and edge_labels is not None:
            self.memory.add(edge_emb, edge_labels)

        return logits


# =====================================================
# Edge labels
# =====================================================
def create_edge_labels(G, labels, edge_index, node_to_idx):
    edge_labels = []
    nodes_list = list(G.nodes())
    for i in range(edge_index.size(1)):
        src_idx = edge_index[0][i].item()
        dst_idx = edge_index[1][i].item()

        src_node = nodes_list[src_idx]
        dst_node = nodes_list[dst_idx]

        src_label = 1 if labels[src_node][0] == 'set_irregular' else 0
        dst_label = 1 if labels[dst_node][0] == 'set_irregular' else 0

        # Edge is labeled as irregular if either endpoint is irregular
        edge_labels.append(float(src_label or dst_label))

    return torch.tensor(edge_labels, dtype=torch.float)


# =====================================================
# Graph construction & node labeling
# =====================================================
def create_graph(data):
    G = nx.DiGraph()
    nodes = set(data['from_address'].tolist() + data['to_address'].tolist())
    G.add_nodes_from(nodes)
    for _, row in data.iterrows():
        G.add_edge(row['from_address'], row['to_address'], weight=row['timestamp'])
    return G


def calculate_fraud_and_antifraud_scores(G):
    fraud_scores = nx.out_degree_centrality(G)
    antifraud_scores = nx.eigenvector_centrality(G, max_iter=10000)
    return fraud_scores, antifraud_scores


def label_nodes(fraud_scores, antifraud_scores,
                fraud_threshold=0.01, antifraud_threshold=0.01):
    labels = {}
    for node in fraud_scores:
        set_label = ('set_irregular'
                            if fraud_scores[node] > fraud_threshold
                            else 'set_regular')
        payment_label = ('payment_regular'
                     if antifraud_scores[node] > antifraud_threshold
                     else 'payment_irregular')
        labels[node] = (set_label, payment_label)
    return labels


# =====================================================
# Reachability features
# =====================================================
def create_reachability_subgraph(G, node, max_depth=1):
    reachability_subgraph = nx.DiGraph()
    reachability_subgraph.add_node(node)
    current_level = {node}
    for depth in range(max_depth):
        next_level = set()
        for u in current_level:
            for v in G.successors(u):
                if v not in reachability_subgraph:
                    reachability_subgraph.add_edge(u, v, weight=G[u][v]['weight'])
                    next_level.add(v)
        current_level = next_level
    return reachability_subgraph


def label_edges(G, max_depth=1):
    reachability_networks = defaultdict(nx.DiGraph)
    for node in G.nodes:
        reachability_networks[node] = create_reachability_subgraph(G, node, max_depth)
    return reachability_networks


def count_edges(reachability_networks, label):
    count = 0
    for u, v, data in reachability_networks.edges(data=True):
        if label in data:
            count += 1
    return count


def common_eval(reachability_networks):
    neighbors = {}
    for node, reach_net in reachability_networks.items():
        neighbors[node] = list(reach_net.neighbors(node))
    return neighbors


def extract_features(G, node):
    reachability_networks = label_edges(G, max_depth=1)
    neighbors = common_eval(reachability_networks)
    S1 = count_edges(reachability_networks[node], label='set_regular')
    S2 = count_edges(reachability_networks[node], label='set_irregular')
    S3 = count_edges(reachability_networks[node], label='payment_regular')
    S4 = count_edges(reachability_networks[node], label='payment_irregular')

    node_features = [S1, S2, len(neighbors.get(node, [])),
                     S3, S4, len(neighbors.get(node, [])),
                     G.in_degree(node), G.out_degree(node)]
    return node_features


# =====================================================
# Dataset builder
# =====================================================
def create_data_list(G_list, labels_list):
    data_list = []
    for G, labels in zip(G_list, labels_list):
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}

        edge_index = torch.tensor(
            [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges],
            dtype=torch.long
        ).t().contiguous()

        x = torch.tensor(
            [extract_features(G, node) for node in G.nodes],
            dtype=torch.float
        )

        edge_labels = create_edge_labels(G, labels, edge_index, node_to_idx)

        data = Data(x=x, edge_index=edge_index, y=edge_labels)
        data_list.append(data)

    return data_list


# =====================================================
# Training (end-to-end with BCE-with-logits; SVM fit removed)
# =====================================================
def train_model(model, train_loader, epochs=200, learning_rate=0.0003):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            logits = model(
                data.x, data.edge_index, data.batch,
                edge_labels=data.y, training=True
            )

            pos_weight = torch.tensor(
                (len(data.y) - data.y.sum()) / (data.y.sum() + 1e-6),
                device=device
            )

            loss = F.binary_cross_entropy_with_logits(
                logits, data.y, pos_weight=pos_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {avg:.4f}")

    print(f"Model trained for {epochs} epochs (MLP classifier, RAG enabled).")
    return model


# =====================================================
# Evaluation
# =====================================================
def evaluate_model(model, loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch,
                           training=False)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_predictions.append(probs)
            all_labels.append(batch.y.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    binary_preds = (all_predictions > 0.5).astype(int)

    auc_score = roc_auc_score(all_labels, all_predictions)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Regular", "Irregular"],
                yticklabels=["Regular", "Irregular"])
    plt.xlabel("Predicted", fontsize=22)
    plt.ylabel("True", fontsize=22)
    plt.title("Confusion Matrix", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate", fontsize=22)
    plt.ylabel("True Positive Rate", fontsize=22)
    plt.title("ROC Curve", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # ---- Precision-Recall Curve ----
    precision_vals, recall_vals, _ = precision_recall_curve(
        all_labels, all_predictions
    )
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='purple')
    plt.xlabel("Recall", fontsize=22)
    plt.ylabel("Precision", fontsize=22)
    plt.title("Precision-Recall Curve", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.show()

    return auc_score, precision, recall, f1


# =====================================================
# Main
# =====================================================
def main():
    # Load the dataset
    file_path = 'soc-sign-bitcoinalpha.csv'
    data = pd.read_csv(file_path)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    # Sort by timestamp
    data = data.sort_values(by='timestamp')

    # 30-day time slices
    data['time_slice'] = (data['timestamp'] - data['timestamp'].min()).dt.days // 30
    '''
    # Collapse sparse trailing slices
    combined_time_slice = max(data['time_slice']) - 1
    data.loc[data['time_slice'] >= combined_time_slice, 'time_slice'] = combined_time_slice

    new_time_slice_counts = data['time_slice'].value_counts().sort_index()
    print(new_time_slice_counts)'''

    threshold = 20
    #time_slice_counts = data['time_slice'].values.counts()
    time_slice_counts = data['time_slice'].value_counts()
    sparse_time_slices = time_slice_counts[time_slice_counts < threshold].index.tolist()

    if sparse_time_slices:
        combined_time_slice = max(data['time_slice']) + 1  
        data.loc[data['time_slice'].isin(sparse_time_slices), 'time_slice'] = combined_time_slice

    data['time_slice'] = data['time_slice'].astype(int)
    new_time_slice_counts = data['time_slice'].value_counts().sort_index()
    print(new_time_slice_counts)

    # Build per-slice graphs and node labels
    G_list = []
    labels_list = []

    for time_slice in data['time_slice'].unique():
        slice_data = data[data['time_slice'] == time_slice]
        G = create_graph(slice_data)
        fraud_scores, antifraud_scores = calculate_fraud_and_antifraud_scores(G)
        labels = label_nodes(fraud_scores, antifraud_scores)
        G_list.append(G)
        labels_list.append(labels)

    # 60/20/20 chronological split
    if len(G_list) > 2:
        train_G, temp_G, train_labels, temp_labels = train_test_split(
            G_list, labels_list, test_size=0.4, shuffle=False
        )
        val_G, test_G, val_labels, test_labels = train_test_split(
            temp_G, temp_labels, test_size=0.5, shuffle=False
        )
    else:
        train_G, train_labels = G_list, labels_list
        val_G, val_labels, test_G, test_labels = [], [], [], []

    # Build Data objects + loaders
    train_data_list = create_data_list(train_G, train_labels)
    val_data_list = create_data_list(val_G, val_labels) if val_G else []
    test_data_list = create_data_list(test_G, test_labels) if test_G else []

    train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False) if val_data_list else None
    test_loader = DataLoader(test_data_list, batch_size=16, shuffle=False) if test_data_list else None

    # Initialize model on device
    model = EnhancedTemporalGraphNetwork(
        input_dim=8,
        hidden_dim=16,
        num_layers=2
    ).to(device)

    # Train
    trained_model = train_model(model, train_loader)

    # Evaluate
    if test_loader:
        auc_score, precision, recall, f1 = evaluate_model(trained_model, test_loader)
        print(f"Test AUC : {auc_score:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
    else:
        print("Not enough data to create a test set.")

    return trained_model


if __name__ == "__main__":
    trained_model = main()
