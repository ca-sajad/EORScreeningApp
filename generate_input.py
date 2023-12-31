"""
generates data samples based on standard deviation and average of  distributions for EOR projects
"""

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from typing import List, Tuple

TRAIN_PORTION = 0.8


class EORDataset(Dataset):
    def __init__(self, data: List[List[float]], labels: List[str]) -> None:
        self.train_data = data
        self.classes = labels
        self.class_to_idx = {name: i for i, name in enumerate(set(self.classes))}

    def __len__(self) -> int:
        return len(self.train_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.train_data[index]
        class_name = self.classes[index]
        class_idx = self.class_to_idx[class_name]

        return torch.tensor(sample, dtype=torch.float32), class_idx


def generate_samples(ave_list: List[List[str | float]], std_dev_list: List[List[str | float]], num_samples: int) -> \
        Tuple[List[List[float]], List[str]]:
    """

    :param ave_list:
    :param std_dev_list:
    :param num_samples:
    :return:
    """
    train_data = []
    train_labels = []
    for l1, l2 in zip(ave_list, std_dev_list):
        train_labels.extend([l1[0]] * num_samples)

        prop1 = np.random.normal(loc=l1[1], scale=l2[1], size=num_samples)
        prop2 = np.random.normal(loc=l1[2], scale=l2[2], size=num_samples)
        prop3 = np.random.normal(loc=l1[3], scale=l2[3], size=num_samples)
        prop4 = np.random.normal(loc=l1[4], scale=l2[4], size=num_samples)
        prop5 = np.random.normal(loc=l1[5], scale=l2[5], size=num_samples)
        prop6 = np.random.normal(loc=l1[6], scale=l2[6], size=num_samples)
        prop7 = np.random.normal(loc=l1[7], scale=l2[7], size=num_samples)

        samples = [[p1, p2, p3, p4, p5, p6, p7] for p1, p2, p3, p4, p5, p6, p7 in
                   zip(prop1, prop2, prop3, prop4, prop5, prop6, prop7)]
        train_data.extend(samples)

    return train_data, train_labels


def calculate_minmax():
    pass


def create_dataset(data: List[List[float]], labels: List[str]) -> EORDataset:
    return EORDataset(data, labels)


def get_train_valid_data(EOR_dataset: EORDataset) -> List[Dataset]:
    return random_split(EOR_dataset, [TRAIN_PORTION, 1-TRAIN_PORTION])

