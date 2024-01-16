import os
import csv
import torch
from torch import nn
from typing import List
from torch.utils.data import DataLoader
from data_utils import load_test_data
from model import EORMulticlassModel
from plots import plot_precision_recall, plot_f1_score, plot_roc_curve
from constants import *


def load_model(model_path: str) -> nn.Module:
    """Loads and returns an EORMulticlassModel

    :param model_path: a str containing model file path
    :return: an EORMulticlassModel
    """
    kwargs, state = torch.load(model_path)
    model = EORMulticlassModel(**kwargs)
    model.load_state_dict(state)

    return model


def run_one_test(model_path: str, input_data: List[float]) -> int:
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


def save_test_results(test_results: List[List[float]]) -> None:
    """Saves the test results in a csv file

    csv file address is read from constants.py

    :param test_results: a 2d list of floats where the elements in each row are
            - the target class
            - the predicted class
            - the probabilities of each class
           Hence, size of each row is 2+NUM_CLASSES
    :return: None
    """
    with open(TEST_RESULTS_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(test_results)


def run_tests(model_path: str) -> float:
    """Runs the set of test data and plots metrics

    This functions reads test data from the Excel file determined in constants.py,
    runs the model saved in training and entered as function input, calculates and
    plots metrics.
    The metrics include f1-score, precision-recall, and Receiver Operating Characteristic (ROC).

    :param model_path: the path of the model to be run
    :return: test accuracy
    """
    model = load_model(model_path=model_path)

    test_dataset = load_test_data()

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 num_workers=os.cpu_count())
    test_acc = 0
    test_results = []
    f1_preds = torch.empty(0)
    target = torch.empty(0)
    pr_preds = torch.empty(0)

    model.eval()

    with torch.inference_mode():
        for inputs, classes in test_dataloader:
            test_pred_logits = model(inputs)
            calculated_classes = test_pred_logits.argmax(dim=1)
            test_acc += ((calculated_classes == classes).sum().item() / len(calculated_classes))
            # collect metrics to be plotted
            f1_preds = torch.cat((f1_preds, calculated_classes), dim=0)
            target = torch.cat((target, classes), dim=0)
            pr_preds = torch.cat((pr_preds, test_pred_logits), dim=0)
            # collect results to be saved in csv file
            results = torch.cat((torch.unsqueeze(classes, dim=1),
                                 torch.unsqueeze(calculated_classes, dim=1),
                                 torch.softmax(test_pred_logits, dim=1)),
                                dim=1)
            test_results.extend(results.tolist())

    test_acc /= len(test_dataloader)
    # save test targets and predictions to a csv file
    save_test_results(test_results=test_results)
    # plot f1 score for all classes
    plot_f1_score(f1_preds, target)
    # plot ROC curve
    plot_roc_curve(pr_preds, target.long())
    # plot precision-recall curve
    plot_precision_recall(pr_preds, target.long())

    return test_acc
