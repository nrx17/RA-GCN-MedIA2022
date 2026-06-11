import os
import pickle
import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import cosine_similarity


N = 1000
GRAPH_FEATURES = 10
NODE_FEATURES = 10
TOTAL_FEATURES = GRAPH_FEATURES + NODE_FEATURES

MAJORITY_RATIO = 0.90
THRESHOLD = 0.5

n_major = int(N * MAJORITY_RATIO)
n_minor = N - n_major

X, y = make_classification(
    n_samples=N,
    n_features=TOTAL_FEATURES,
    n_informative=16,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=[MAJORITY_RATIO, 1 - MAJORITY_RATIO],
    class_sep=1.5,
    random_state=17,
)

graph_features = X[:, :GRAPH_FEATURES]
node_features = X[:, GRAPH_FEATURES:]

sim = cosine_similarity(graph_features)

adj = (sim >= THRESHOLD).astype(np.float32)
np.fill_diagonal(adj, 0)

os.makedirs("data/synthetic", exist_ok=True)

with open("data/synthetic/per-90gt-0.5.pkl", "wb") as f:
    pickle.dump((adj, node_features, y), f)

print("saved")
print(adj.shape)
print(node_features.shape)
print(y.shape)
print("edges:", int(adj.sum() / 2))