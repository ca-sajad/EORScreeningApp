"""functions for plotting EOR data

"""
from itertools import accumulate
from typing import List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def scatter_plot(data: List[List[float]], labels: List[str]) -> None:
    """Creates a scatter plot of the first two PCs of EOR data

    This function transforms a 2d list of oilfield properties to their first two
    principal components and shows a scatter plot of these PCs.

    :param data: a 2d list of oilfield properties of each data sample
    :param labels: a 1d list of EOR method corresponding to that data sample
    :return: None
    """
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
    """ Creates a line plot of cumulative values of PC variance ratios

    This function calculates explained_variance_ratio of principal components
    and shows a line plot of cumulative values of these ratios.

    :param data: a 2d list of oilfield properties of each data sample
    :return: None
    """
    pca = PCA(n_components=7)
    pca_data = pca.fit(data)
    variance_ratios = pca_data.explained_variance_ratio_
    cumulative_variances = list(accumulate(variance_ratios))
    plt.plot(cumulative_variances)
    plt.title('Plot of pca variance ratios')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()