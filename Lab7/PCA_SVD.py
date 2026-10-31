# Lab 7: PCA and SVD – Dimensionality Reduction

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Original Dataset Shape:", X.shape)

# 2. Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA Dataset Shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# 4. Visualize PCA Results
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Iris Dataset")
plt.savefig("pca_output.png")
plt.show()

# 5. Apply Singular Value Decomposition (SVD)
U, S, VT = np.linalg.svd(X_scaled, full_matrices=False)

print("U Matrix Shape:", U.shape)
print("Singular Values:", S)
print("VT Matrix Shape:", VT.shape)

# 6. Reconstruct Data using SVD
k = 2
X_reconstructed = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

print("Original Data Shape:", X_scaled.shape)
print("Reconstructed Data Shape:", X_reconstructed.shape)

# 7. Display Results
plt.figure(figsize=(8, 5))
plt.plot(S, marker="o")
plt.title("Singular Values using SVD")
plt.xlabel("Component")
plt.ylabel("Singular Value")
plt.savefig("svd_output.png")
plt.show()

print("PCA and SVD Completed Successfully!")
