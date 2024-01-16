"""Contains functions to normalize EOR_data and generate pytorch datasets
"""

import numpy as np
import json
from typing_extensions import override
import torch
from torch.utils.data import Dataset, random_split
from typing import List, Tuple, Dict
from constants import *
from extract_data import read_distribution_file, read_test_dataset_file


class EORDataset(Dataset):
    """a pytorch Dataset containing oilfield properties data and their labels
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
            prop = lmin[i] + (lmax[i] - lmin[i]) * np.random.normal(loc=0.5, scale=0.18, size=samples_per_class)
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


def read_mins_maxs(file: str, labels: List[str]) -> Tuple[List[float], List[float]]:
    """Reads minimum and maximum values for each property from a json file

    :param file: the path of a json file containing minimum and maximum values
    :param labels: a list of (oilfield) properties whose min and max are to be read
    :return: a tuple where the first item is a list of minimum values of properties
             and the second item is a list of maximum values of properties
    """
    with open(file, 'r') as json_file:
        data = json.load(json_file)

    mins = []
    maxs = []
    for label in labels:
        mins.append(data[label]['min'])
        maxs.append(data[label]['max'])

    return mins, maxs


def load_input_data() -> EORDataset:
    """Reads data from an Excel file and create an input(train/validation) dataset

    :return: an EORDataset containing the combined train and validation dataset
    """
    # read minimum and maximum of EOR properties,
    # input_data_dict is in this form: {'min_list':list, 'max_list':list, 'prop_labels':list}
    input_data_dict = read_distribution_file()
    # create input data
    input_data, input_labels = generate_samples(data_dict=input_data_dict,
                                                samples_per_class=SAMPLES_PER_CLASS,
                                                props_count=INPUT_SIZE)
    # calculate minimum and maximum of generated input_data
    min_properties, max_properties = calculate_minmax(input_data)
    # save mins and maxs to a json file
    save_mins_maxs(mins=min_properties,
                   maxs=max_properties,
                   labels=input_data_dict['prop_labels'],
                   file=MIN_MAX_FILE)
    # normalize data between 0 and 1
    norm_input_data = normalize_data(mins=min_properties,
                                     maxs=max_properties,
                                     data=input_data)
    # plot input using first two Principal Components
    # scatter_plot(norm_input_data, input_labels)
    # create input dataset
    input_dataset = create_dataset(data=norm_input_data, labels=input_labels)

    return input_dataset


def load_test_data() -> EORDataset:
    """Reads data from an Excel file and create a test dataset

    :return: an EORDataset containing the test dataset
    """
    # get test data
    # test_data_dict is in this form: {'sample_data':list, 'sample_label':list, 'prop_labels':list}
    test_data_dict = read_test_dataset_file()
    # reading min and max of properties
    min_properties, max_properties = read_mins_maxs(file=MIN_MAX_FILE, labels=test_data_dict['prop_labels'])

    # normalize data between 0 and 1
    norm_test_data = normalize_data(mins=min_properties,
                                    maxs=max_properties,
                                    data=test_data_dict['sample_data'])
    # create test dataset
    test_dataset = create_dataset(data=norm_test_data, labels=test_data_dict['sample_label'])

    return test_dataset


def load_data() -> Tuple[EORDataset, EORDataset]:
    """Loads both input(train/validation) and test datasets

    :return: a Tuple where the first element is input dataset and the second element
             is testing dataset
    """
    input_dataset = load_input_data()
    test_dataset = load_test_data()

    return input_dataset, test_dataset
