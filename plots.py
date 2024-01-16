"""functions for plotting EOR data

"""
from itertools import accumulate
from typing import List
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassPrecisionRecallCurve, MulticlassF1Score, MulticlassROC
from constants import NUM_CLASSES


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


def plot_f1_score(preds: torch.Tensor, target: torch.Tensor) -> None:
    """Plots f1 score of all classes

    :param preds: a 1d tensor of model predictions whose length is the number of samples
    :param target: a 1d tensor of actual values whose length is the number of samples
    :return: None
    """
    mcf1s = MulticlassF1Score(num_classes=NUM_CLASSES, average=None)
    mcf1s.update(preds, target)
    fig_, ax_ = mcf1s.plot()


def plot_precision_recall(preds: torch.Tensor, target: torch.Tensor) -> None:
    """Plots precision-recall curve for each class

    This function also shows the area under curve (AUC)

    :param preds: a 2d tensor of logits of model predictions of size
                    (number of samples) x (number of classes)
    :param target: a 1d tensor of actual values whose length is the number of samples
    :return: None
    """
    metric = MulticlassPrecisionRecallCurve(num_classes=NUM_CLASSES, thresholds=None)
    metric.update(preds, target)
    fig_, ax_ = metric.plot(score=True)


def plot_roc_curve(preds: torch.Tensor, target: torch.Tensor) -> None:
    """Plots Receiver Operating Characteristic (ROC) curve for each class

    This function also shows the area under curve (AUC)

    :param preds: a 2d tensor of logits of model predictions of size
                    (number of samples) x (number of classes)
    :param target: a 1d tensor of actual values whose length is the number of samples
    :return: None
    """
    mcf1s = MulticlassROC(num_classes=NUM_CLASSES, thresholds=None)
    mcf1s.update(preds, target)
    fig_, ax_ = mcf1s.plot(score=True)
