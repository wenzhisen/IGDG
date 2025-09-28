import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def mutual_information(x: torch.Tensor, y: torch.Tensor, k: int = 3) -> float:
    """
    Calculate the mutual information between two tensors using the k-nearest neighbor approach.

    Parameters:
    x (torch.Tensor): Tensor of shape (n_samples, n_features_x).
    y (torch.Tensor): Tensor of shape (n_samples, n_features_y).
    k (int): Number of nearest neighbors to use in the MI estimation. Default is 3.

    Returns:
    float: Estimated mutual information between x and y.
    """
    # Convert tensors to numpy arrays for compatibility with sklearn
    x = x.cpu().numpy()
    y = y.cpu().numpy()

    # Number of samples
    n_samples = x.shape[0]

    # Combine x and y to calculate distances in the joint space
    xy = np.concatenate([x, y], axis=1)

    # Fit nearest neighbors in the joint space
    nn_xy = NearestNeighbors(metric='chebyshev', n_neighbors=k + 1).fit(xy)
    nn_x = NearestNeighbors(metric='chebyshev', n_neighbors=k + 1).fit(x)
    nn_y = NearestNeighbors(metric='chebyshev', n_neighbors=k + 1).fit(y)

    # Find the k-th nearest neighbor distance in the joint space
    distances_xy, _ = nn_xy.kneighbors(xy)
    distances_x, _ = nn_x.kneighbors(x)
    distances_y, _ = nn_y.kneighbors(y)

    # Get the k-th nearest distance
    k_distance_xy = distances_xy[:, k]
    k_distance_x = distances_x[:, k]
    k_distance_y = distances_y[:, k]

    # Count neighbors within the k-th nearest neighbor distance
    n_x = np.sum(distances_x <= np.expand_dims(k_distance_xy, axis=1), axis=1) - 1
    n_y = np.sum(distances_y <= np.expand_dims(k_distance_xy, axis=1), axis=1) - 1

    # Calculate the mutual information
    mi_estimate = (np.log(n_samples) + np.log(k) - np.mean(np.log(n_x + 1) + np.log(n_y + 1)))

    return mi_estimate
n_samples = 100
x = torch.randn(n_samples, 2)  # 从标准正态分布中生成 x 的采样
y = x + 0.5 * torch.randn(n_samples, 2)  # y 是 x 加上小的噪声，因此 x 和 y 是相关的
# 计算互信息
print(torch.cuda.is_available())
mi = mutual_information(x, y, k=3)
print(f"Estimated Mutual Information: {mi}")