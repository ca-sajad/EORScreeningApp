
import os
import torch
from torch import nn
from typing import List
from torch.utils.data import DataLoader
from model_utils import run_model
from data_utils import load_data, load_test_data
from model import EORMulticlassModel


def train_once() -> None:
    hyper_params = {
        'hidden_size': 32,
        'batch_size': 32,
        'learning_rate': 0.1,
        'num_epochs': 200,
        'train_portion': 0.8,
    }

    input_dataset, test_dataset = load_data()
    result_dict = run_model(input_dataset=input_dataset,
                            test_dataset=test_dataset,
                            params=hyper_params)
    print(f"test accuracy: {result_dict['test_acc']}")


def load_model(model_path: str) -> nn.Module:
    """Loads and returns an EORMulticlassModel

    :param model_path: a str containing model file path
    :return: an EORMulticlassModel
    """
    kwargs, state = torch.load(model_path)
    model = EORMulticlassModel(**kwargs)
    return model.load_state_dict(state)


def test_single_input(model_path: str, input_data: List[float]) -> int:
    """
    Test a single input with the model and return the predicted class

    :param model_path: the path of an EORMulticlassModel.
    :param input_data: The input data to be tested (a single sample).
    :return: The predicted class index.
    """
    model = load_model(model_path=model_path)
    sample = torch.tensor(input_data, dtype=torch.float32)

    model.eval()
    with torch.inference_mode():
        output = model(sample)
        _, predicted = torch.max(output, 1)

    return predicted.item()


def run_save_test_results(model_path: str):
    model = load_model(model_path=model_path)
    test_dataset = load_test_data()

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 num_workers=os.cpu_count())
    test_acc = 0
    model.eval()

    with torch.inference_mode():
        for inputs, classes in test_dataloader:
            test_pred_logits = model(inputs)
            calculated_classes = test_pred_logits.argmax(dim=1)
            test_acc += ((calculated_classes == classes).sum().item() / len(calculated_classes))

    test_acc /= len(test_dataloader)

    return test_acc