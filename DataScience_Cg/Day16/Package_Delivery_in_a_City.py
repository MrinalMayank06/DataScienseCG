# Scenario: Package Delivery in a City
# A delivery company operates in a large city containing 100,000 houses.
# Each house is represented by a unique numerical fingerprint, similar to
# a high-dimensional embedding vector (for example 384 dimensions).
#
# Problem:
# When a customer places an order, the system must quickly determine the
# nearest warehouse or delivery hub. If the system compares the query
# against every house individually, the process becomes slow for large data.
#
# Solution:
# Use Approximate Nearest Neighbor (ANN) indexing to retrieve the nearest
# vectors efficiently without scanning the entire dataset.
#
# Two ANN strategies are used:
#
# HNSW (Hierarchical Navigable Small World)
# - Builds a graph structure between vectors.
# - Each vector connects to nearby neighbors.
# - During search, the algorithm moves through this graph to reach
#   the closest matches efficiently.
#
# IVF (Inverted File Index)
# - Divides the dataset into clusters (Voronoi cells).
# - At query time, the algorithm searches only within the most relevant
#   clusters rather than the entire dataset.
#
# Goal:
# Build HNSW and IVF indices for 100,000 vectors and retrieve the
# top-10 nearest neighbors for a query vector.



import faiss
import numpy as np


# Step 1: Define dataset parameters
dimension = 384
num_vectors = 100000


# Step 2: Generate synthetic vectors representing house fingerprints
vectors = np.random.randn(num_vectors, dimension).astype("float32")

# Normalize vectors to ensure consistent similarity behavior
faiss.normalize_L2(vectors)


# -------------------------------------------------
# HNSW Index Construction
# -------------------------------------------------

hnsw_index = faiss.IndexHNSWFlat(dimension, 32)

hnsw_index.hnsw.efConstruction = 200
hnsw_index.hnsw.efSearch = 100

hnsw_index.add(vectors)


# Generate a query vector representing a customer location
query = np.random.randn(1, dimension).astype("float32")
faiss.normalize_L2(query)

distances_hnsw, indices_hnsw = hnsw_index.search(query, 10)

print("HNSW Nearest Indices:", indices_hnsw[0])
print("HNSW Distances:", distances_hnsw[0])


# -------------------------------------------------
# IVF Index Construction
# -------------------------------------------------

num_clusters = 256

quantizer = faiss.IndexFlatL2(dimension)

ivf_index = faiss.IndexIVFFlat(quantizer, dimension, num_clusters)

ivf_index.train(vectors)

ivf_index.add(vectors)

ivf_index.nprobe = 10

distances_ivf, indices_ivf = ivf_index.search(query, 10)

print("IVF Nearest Indices:", indices_ivf[0])
print("IVF Distances:", distances_ivf[0])