import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import Linear, Dropout
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATv2Conv
import visualization
from torch_geometric.utils.convert import from_networkx, to_networkx
import random

class GCN(torch.nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        return h, F.log_softmax(h, dim=1)


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h * 4, heads=heads)
        self.gat2 = GATv2Conv(dim_h * 8 * 4, dim_h, heads=4)
        self.gat3 = GATv2Conv(dim_h * 4, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat3(h, edge_index)
        return h, F.log_softmax(h, dim=1)


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


def train(model, data):
    """Train a GNN model and return the trained model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 100

    model.train()
    for epoch in range(epochs + 1):
        # Training
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        # Print metrics every 10 epochs
        if (epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc * 100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc * 100:.2f}%')

    return model


def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    visualization.visualize_confusion_cora(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask])
    visualization.visualize_confusion_cora(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask])
    return acc


# Import dataset from PyTorch Geometric
dataset = Planetoid(root=".", name="Cora")
data = dataset[0]

original_graph = to_networkx(dataset[0]).to_undirected()
# Create a copy of the original citation graph
augmented_graph = original_graph.copy()

# Edge addition
num_edges_to_add = 500
for _ in range(num_edges_to_add):
    node_a, node_b = random.sample(list(original_graph.nodes()), 2)
    augmented_graph.add_edge(node_a, node_b)

# Edge removal
num_edges_to_remove = 100
edges_to_remove = random.sample(list(augmented_graph.edges()), num_edges_to_remove)
augmented_graph.remove_edges_from(edges_to_remove)

augmented_data = from_networkx(augmented_graph)

data.edge_index = augmented_data.edge_index

# Create GCN
gcn = GCN(dataset.num_features, 14, dataset.num_classes)
print(gcn)

# Train
train(gcn, data)

# Test
acc = test(gcn, data)
print(f'GCN test accuracy: {acc * 100:.2f}%\n')

# Create GAT
gat = GAT(dataset.num_features, 16, dataset.num_classes)
print(gat)

# Train
train(gat, data)

# Test
acc = test(gat, data)
print(f'GAT test accuracy: {acc * 100:.2f}%\n')

untrained_gat = GAT(dataset.num_features, 16, dataset.num_classes)

# Get embeddings
h, _ = untrained_gat(data.x, data.edge_index)

# Train TSNE
tsne = TSNE(n_components=2, learning_rate='auto',
            init='pca').fit_transform(h.detach())

# Plot TSNE
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
plt.show()

h, _ = gat(data.x, data.edge_index)

# Train TSNE
tsne = TSNE(n_components=2, learning_rate='auto',
            init='pca').fit_transform(h.detach())

# Plot TSNE
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
plt.show()
