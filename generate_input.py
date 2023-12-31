"""
generates data samples based on standard deviation and average of  distributions for EOR projects
"""

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from typing import List, Tuple
from sklearn.decomposition import PCA


TRAIN_PORTION = 0.8
NUM_CLASSES = 9
NUM_PROPERTIES = 7

class EORDataset(Dataset):
    def __init__(self, data: List[List[float]], labels: List[str]) -> None:
        self.train_data = data
        self.classes = labels
        sorted_set = sorted(set(self.classes), key=lambda s: s.lower())
        self.class_to_idx = {name: i for i, name in enumerate(sorted_set)}

    def __len__(self) -> int:
        return len(self.train_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.train_data[index]
        class_name = self.classes[index]
        class_idx = self.class_to_idx[class_name]

        return torch.tensor(sample, dtype=torch.float32), class_idx


def generate_samples(ave_list: List[List[str | float]], std_dev_list: List[List[str | float]],
                     min_list: List[List[str | float]], max_list: List[List[str | float]],
                     num_samples: int) -> Tuple[List[List[float]], List[str]]:
    """

    :param ave_list:
    :param std_dev_list:
    :param num_samples:
    :return:
    """
    input_data = []
    input_labels = []
    additional_samples = 10 * num_samples

    for l1, l2, l3, l4 in zip(ave_list, std_dev_list, min_list, max_list):
        input_labels.extend([l1[0]] * num_samples)
        samples = []

        props = []
        for i in range(1, NUM_PROPERTIES+1):
            prop = np.random.normal(loc=l1[i], scale=l2[i]*0.5, size=additional_samples)
            prop = list(filter(lambda p: l3[i] <= p <= l4[i], prop))
            props.append(prop[:num_samples])

        for i in range(num_samples):
            sample = []
            for j in range(NUM_PROPERTIES):
                sample.append(props[j][i])
            samples.append(sample)

        input_data.extend(samples)

    return input_data, input_labels


def calculate_minmax():
    pass


def create_dataset(data: List[List[float]], labels: List[str]) -> EORDataset:
    return EORDataset(data, labels)


def get_train_valid_data(EOR_dataset: EORDataset) -> List[Dataset]:
    print(EOR_dataset.class_to_idx)
    return random_split(EOR_dataset, [TRAIN_PORTION, 1 - TRAIN_PORTION])


def convert_to_PCA(data: List[List[float]]) -> List[List[float]]:
    pca = PCA(n_components=2)
    return pca.fit_transform(data)
