"""
generates data samples based on standard deviation and average of  distributions for EOR projects
"""

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from typing import List, Tuple
from sklearn.decomposition import PCA

PCA_COMPONENTS = 4
TRAIN_PORTION = 0.8
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

    input_data = []
    input_labels = []

    ############ temporarily delete all classes except N2 and steam
    min_list = [l for i, l in enumerate(min_list) if i == 0 or i == 8]
    max_list = [l for i, l in enumerate(max_list) if i == 0 or i == 8]

    # generating samples using min and max
    for lmin, lmax in zip(min_list, max_list):
        input_labels.extend([lmin[0]] * num_samples)
        props = []
        for i in range(1, NUM_PROPERTIES+1):
            prop = lmin[i] + (lmax[i] - lmin[i]) * np.random.normal(loc=0.5, scale=0.18, size=num_samples)
            props.append(prop)
        samples = [[props[j][i] for j in range(NUM_PROPERTIES)] for i in range(num_samples)]
        input_data.extend(samples)

    # generating samples using average and std dev
    # additional_samples = 10 * num_samples
    # for l1, l2, l3, l4 in zip(ave_list, std_dev_list, min_list, max_list):
    #     input_labels.extend([l1[0]] * num_samples)
    #     props = []
    #     for i in range(1, NUM_PROPERTIES+1):
    #         prop = np.random.normal(loc=l1[i], scale=l2[i]*0.5, size=additional_samples)
    #         prop = list(filter(lambda p: l3[i] <= p <= l4[i], prop))
    #         props.append(prop[:num_samples])
    #     samples = [[props[j][i] for j in range(NUM_PROPERTIES)] for i in range(num_samples)]
    #     input_data.extend(samples)

    return input_data, input_labels


def calculate_minmax(data: List[List[float]]) -> Tuple[List[float], List[float]]:
    maxs = [-1] * NUM_PROPERTIES
    mins = [10000] * NUM_PROPERTIES

    for sample in data:
        maxs = [sample[i] if sample[i] > maxs[i] else maxs[i] for i in range(len(sample))]
        mins = [sample[i] if sample[i] < mins[i] else mins[i] for i in range(len(sample))]

    return maxs, mins


def create_dataset(data: List[List[float]], labels: List[str]) -> EORDataset:
    return EORDataset(data, labels)


def get_train_valid_data(EOR_dataset: EORDataset) -> List[Dataset]:
    print(EOR_dataset.class_to_idx)
    return random_split(EOR_dataset, [TRAIN_PORTION, 1 - TRAIN_PORTION])


def convert_to_PCA(data: List[List[float]]) -> List[List[float]]:
    pca = PCA(n_components=PCA_COMPONENTS)
    return pca.fit_transform(data)


def normalize_data(data: List[List[float]]) -> List[List[float]]:
    maxs, mins = calculate_minmax(data)
    return [[(sample[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(NUM_PROPERTIES)] for sample in data]