"""Contains functions to normalize EOR_data and generate pytorch datasets
"""

import numpy as np
import json
from typing_extensions import override
import torch
from torch.utils.data import Dataset, random_split
from typing import List, Tuple, Dict


class EORDataset(Dataset):
    """Creates a pytorch Dataset

    This dataset contains oilfield properties data and their labels
    """

    def __init__(self, data: List[List[float]], labels: List[str]) -> None:
        """Initializes the instance based on EOR properties data and labels

        :param
            data: a 2d list where each row is a list of oilfield properties for each EOR method
        :param
            labels: a 1d list of the names of oilfield properties
        """
        self.train_data = data
        self.classes = labels
        sorted_set = sorted(set(self.classes), key=lambda s: s.lower())
        self.class_to_idx = {name: i for i, name in enumerate(sorted_set)}

    @override
    def __len__(self) -> int:
        return len(self.train_data)

    @override
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.train_data[index]
        class_name = self.classes[index]
        class_idx = self.class_to_idx[class_name]

        return torch.tensor(sample, dtype=torch.float32), class_idx


def generate_samples(data_dict: Dict[str, List[List[str | float] | str]],
                     samples_per_class: int,
                     props_count: int) -> Tuple[List[List[float]], List[str]]:
    """Creates a list of data samples for each EOR method

    Creates samples_per_class number of EOR data samples for each class. Each data sample
    includes props_count number of properties. Each property is randomly selected from a
    uniform distribution between minimum and maximum values of that property. For each
    data sample, its corresponding EOR method is added to a list of labels.
    For example, if there are 9 EOR methods, 7 oilfield properties, and 100 samples for each
    method, the labels will be a 900x1 list, and sample data a 900x7 list.

    :param
        data_dict: a dictionary containing
            - key: min_list, value: a 2d list where each element contains minimum values of properties for an EOR method,
                e.g. [['chemical', 0.5, 2.0, 10.0], ['co2_miscible', 0.3, 1.1, 14.0]]
            - key: max_list, value: a 2d list where each element contains maximum values of properties for an EOR method,
                e.g. [['chemical', 2.5, 4.0, 30.0], ['co2_miscible', 0.9, 2.3, 40.0]]
            - key: prop_labels, value: a 1d list of property names
    :param
        samples_per_class: number of samples to be generated for each EOR method
    :param
        props_count: number of properties of each EOR method
    :return:
        a 1d list of EOR samples
        a 1d list of EOR methods
    """

    input_data = []
    input_labels = []

    for lmin, lmax in zip(data_dict['min_list'], data_dict['max_list']):
        input_labels.extend([lmin[0]] * samples_per_class)
        props = []
        for i in range(1, props_count+1):
            prop = lmin[i] + (lmax[i] - lmin[i]) * np.random.normal(loc=0.5, scale=0.2, size=samples_per_class)
            props.append(prop)
        samples = [[props[j][i] for j in range(props_count)] for i in range(samples_per_class)]
        input_data.extend(samples)

    return input_data, input_labels


def calculate_minmax(data: List[List[float]]) -> Tuple[List[float], List[float]]:
    """ Calculates minimum and maximum of parameters

    :param data: a 2d list
    :return:
        a 1d list of minimum values
        a 1d list of maximum values
    """

    props_count = len(data[0])
    maxs = [-1] * props_count
    mins = [10000] * props_count

    for sample in data:
        maxs = [sample[i] if sample[i] > maxs[i] else maxs[i] for i in range(len(sample))]
        mins = [sample[i] if sample[i] < mins[i] else mins[i] for i in range(len(sample))]

    return maxs, mins


def create_dataset(data: List[List[float]], labels: List[str]) -> EORDataset:
    """Creates an EORDataset

    :param data: a 2d list of EOR properties, each row belongs to an oilfield
    :param labels: a 1d list of EOR methods corresponding to data
    :return:
        an EORDataset
    """
    return EORDataset(data, labels)


def get_train_valid_data(EOR_dataset: Dataset, train_portion: float) -> List[Dataset]:
    """Splits an EORDataset into training and validation samples

    :param EOR_dataset: a Dataset
    :param train_portion: ratio of data to be assigned to training, between 0 and 1
    :return:
        a list of Dataset
    """
    return random_split(dataset=EOR_dataset, lengths=[train_portion, 1 - train_portion])


def normalize_data(mins: List[float],
                   maxs: List[float],
                   data: List[List[float]]) -> List[List[float]]:
    """Normalizes a 2d list of data between 0 and 1

    Normalization uses minimum and maximum values of properties

    :param mins: a 1d list of minimum values of properties
    :param maxs: a 1d list of maximum values of properties
    :param data: a 2d list to be normalized
    :return:
        a 2d list of normalized floats between 0 and 1
    """
    props_count = len(data[0])
    return [[(sample[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(props_count)] for sample in data]


def save_mins_maxs(mins: List[float], maxs: List[float], labels: List[str], file: str) -> None:
    """Save minimum and maximum values of each parameter to a json file

    :param mins: a list of minimum values for each parameter used
    :param maxs: a list of maximum values for each parameter used
    :param labels: a list of parameter names
    :param file: file to receive data
    :return:
        None
    """
    data = {}
    for i in range(len(mins)):
        data[labels[i]] = {'min': mins[i], 'max': maxs[i]}

    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
