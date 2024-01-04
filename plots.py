from itertools import accumulate
from typing import List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def scatter_plot(data: List[List[float]], labels: List[str]) -> None:
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    set_labels = set(labels)
    for ind, label in enumerate(set_labels):
        labels = [ind if l == label else l for l in labels]

    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='jet')
    plt.colorbar(scatter, ticks=range(len(set_labels)), label='Classes').set_ticklabels(set_labels)
    plt.title('Scatter Plot of PCA-Reduced Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


def plot_pca(data: List[List[float]]) -> None:
    pca = PCA(n_components=7)
    pca_data = pca.fit(data)
    variance_ratios = pca_data.explained_variance_ratio_
    cumulative_variances = list(accumulate(variance_ratios))
    plt.plot(cumulative_variances)
    plt.title('Plot of pca variance ratios')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()