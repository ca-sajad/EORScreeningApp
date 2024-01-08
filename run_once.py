
from model_utils import run_model, load_data
from model import EORMulticlassModel
import torch
from typing import List

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


def test_single_input(model: EORMulticlassModel, input_data: List[float]) -> int:
    """
    Test a single input with the model and return the predicted class

    :param model: An instance of the EORMulticlassModel.
    :param input_data: The input data to be tested (a single sample).
    :return: The predicted class index.
    """

    sample = torch.tensor(input_data, dtype=torch.float32)

    model.eval()
    with torch.inference_mode():
        output = model(sample)
        _, predicted = torch.max(output, 1)

    return predicted.item()
